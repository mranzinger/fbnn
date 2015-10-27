-- Copyright 2004-present Facebook. All Rights Reserved.

local pl = require('pl.import_into')()

-- from fblualib/fb/util/data.lua , copied here because fblualib is not rockspec ready yet.
-- deepcopy routine that assumes the presence of a 'clone' method in user
-- data should be used to deeply copy. This matches the behavior of Torch
-- tensors.
local function deepcopy(x)
    local typename = type(x)
    if typename == "userdata" then
        return x:clone()
    end
    if typename == "table" then
        local retval = { }
        for k,v in pairs(x) do
            retval[deepcopy(k)] = deepcopy(v)
        end
        return retval
    end
    return x
end

local Optim, parent = torch.class('nn.Optim')


-- Returns weight parameters and bias parameters and associated grad parameters
-- for this module. Annotates the return values with flag marking parameter set
-- as bias parameters set
function Optim.weight_bias_parameters(module)
    local weight_params, bias_params
    if module.weight then
        weight_params = {module.weight, module.gradWeight}
        weight_params.is_bias = false
    end
    if module.bias then
        bias_params = {module.bias, module.gradBias}
        bias_params.is_bias = true
    end
    return {weight_params, bias_params}
end

function Optim.get_module_key(module)

    --return module
    return module.sharedWith or module


    -- Basically, the goal is to understand whether
    -- or not this module shares parameters with another
    -- module. Simplest way to do this is to use the storage
    -- for the weights if the weights exist
    --if not module.weight then
    --    print('No weights')
    --    print(module)
    --    return module
    --end

    --print('Weights')
    --return module.weight:storage()

end

-- The regular `optim` package relies on `getParameters`, which is a
-- beastly abomination before all. This `optim` package uses separate
-- optim state for each submodule of a `nn.Module`.
function Optim:__init(optState)
    
    self.modulesToOptState = {}
    -- Keep this around so we update it in setParameters
    self.originalOptState = optState

    --if not checkpoint_data then
    --else
    --    local state = checkpoint_data.optim_state
    --    local modules = {}
    --    self.model:for_each(function(m) table.insert(modules, m) end)
    --    assert(pl.tablex.compare_no_order(modules, pl.tablex.keys(state)))
    --    self.modulesToOptState = state
    --end
end

function Optim:addModel(model)

    print('Adding model to optimizer')
    print(model)

    local optState = self.originalOptState
    
    -- Each module has some set of parameters and grad parameters. Since
    -- they may be allocated discontinuously, we need separate optState for
    -- each parameter tensor. self.modulesToOptState maps each module to
    -- a lua table of optState clones.
    model:for_each(
        function(module)

            local mod_key = self.get_module_key(module)

            if self.modulesToOptState[mod_key] then
                print('Detected shared module')
                print(module)
            else
                self.modulesToOptState[mod_key] = { }
                local params = self.weight_bias_parameters(module)
                -- expects either an empty table or 2 element table, one for weights
                -- and one for biases
                assert(pl.tablex.size(params) == 0 or pl.tablex.size(params) == 2)
                for i, _ in ipairs(params) do
                    self.modulesToOptState[mod_key][i] = deepcopy(optState)
                    if params[i] and params[i].is_bias then
                        -- never regularize biases
                        self.modulesToOptState[mod_key][i].weightDecay = 0.0
                    end
                end
                assert(module)
                assert(self.modulesToOptState[mod_key])
            end
        end
    )

end

function Optim:save()
    return {
        optim_state = self.modulesToOptState
    }
end

local function _type_all(obj, t)
    for k, v in pairs(obj) do
        if type(v) == 'table' then
            _type_all(v, t)
        else
            local tn = torch.typename(v)
            if tn and tn:find('torch%..+Tensor') then
                obj[k] = v:type(t)
            end
        end
    end
end

function Optim:type(t)
    for k,v in pairs(self.modulesToOptState) do
        _type_all(v, t)
    end
    
    --self.model:for_each(function(module)
    --    local state= self.modulesToOptState[module]
    --    assert(state)
    --    _type_all(state, t)
    --end)
end

local function get_device_for_module(mod)
   local dev_id = nil
   for name, val in pairs(mod) do
       if torch.typename(val) == 'torch.CudaTensor' then
           local this_dev = val:getDevice()
           if this_dev ~= 0 then
               -- _make sure the tensors are allocated consistently
               assert(dev_id == nil or dev_id == this_dev)
               dev_id = this_dev
           end
       end
   end
   return dev_id -- _may still be zero if none are allocated.
end

local function on_device_for_module(mod, f)
    local this_dev = get_device_for_module(mod)
    if this_dev ~= nil then
        return cutorch.withDevice(this_dev, f)
    end
    return f()
end

function Optim:optimize(model, optimMethod, inputs, targets, criterion)
    assert(optimMethod)
    assert(inputs)
    assert(targets)
    assert(criterion)
    assert(self.modulesToOptState)

    model:zeroGradParameters()
    local output = model:forward(inputs)

    local err = criterion:forward(output, targets)

    local df_do = criterion:backward(output, targets)
    model:backward(inputs, df_do)

    -- We'll set these in the loop that iterates over each module. Get them
    -- out here to be captured.
    local curGrad
    local curParam
    local function fEvalMod(x)
        return err, curGrad
    end

    model:for_each(
        function(curMod)
            local mod_key = self.get_module_key(curMod)

            assert(self.modulesToOptState[mod_key],
                   'The specified model hasn\'t been added to the optimizer!')

            local curModParams = self.weight_bias_parameters(curMod)

            -- expects either an empty table or 2 element table, one for weights
            -- and one for biases
            assert(pl.tablex.size(curModParams) == 0 or
                   pl.tablex.size(curModParams) == 2)
            if curModParams then
                for i, tensor in ipairs(curModParams) do
                    if curModParams[i] then
                        -- expect param, gradParam pair
                        curParam, curGrad = table.unpack(curModParams[i])
                        assert(curParam and curGrad)
                        optimMethod(fEvalMod, curParam, opt[i])
                    end
                end
            end

        end
    )

    return err, output
end

function Optim:setParameters(newParams)
    assert(newParams)
    assert(type(newParams) == 'table')
    local function splice(dest, src)
        for k,v in pairs(src) do
            dest[k] = v
        end
    end

    splice(self.originalOptState, newParams)
    for _,optStates in pairs(self.modulesToOptState) do
        for i,optState in pairs(optStates) do
            assert(type(optState) == 'table')
            splice(optState, newParams)
        end
    end
end
