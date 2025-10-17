# Gherkin Feature Files for gguf_tools_mk2

This directory contains Behavior-Driven Development (BDD) feature files written in Gherkin syntax that specify the expected behavior of the model loading classes.

## Overview

The feature files describe the behavior of the model loading infrastructure in a human-readable format. They can be used for:

1. **Documentation** - Understanding how each model class should behave
2. **Test Specifications** - Reference for implementing automated tests
3. **Communication** - Sharing requirements between developers, testers, and stakeholders
4. **Test Implementation** - Can be used with BDD frameworks like `behave` or `pytest-bdd`

## Feature Files

### model_classes/

#### 1. `model_protocol.feature`
Specifies the common interface that all model classes must implement:
- Required methods and their signatures
- Consistent error handling behavior
- Polymorphic usage patterns
- Type hint compatibility

#### 2. `gguf_model.feature`
Specifies behavior for the GGUFModel class:
- Loading GGUF model files
- Tensor name retrieval
- Tensor validation (existence, type, dimensions)
- Data retrieval with type conversion (F16, F32, Q8_0)
- Error handling for missing dependencies and invalid tensors

#### 3. `torch_model.feature`
Specifies behavior for the TorchModel class:
- Loading PyTorch model files
- CPU mapping and memory-mapped loading
- Tensor squeezing behavior
- Tensor validation (dtype, dimensions)
- Data conversion to float32 and numpy
- Error handling

#### 4. `safetensor_model.feature`
Specifies behavior for the SafetensorModel class:
- Loading Safetensor model files
- Lazy loading and memory efficiency
- Metadata handling
- Tensor validation (multiple dtypes, dimensions)
- Data conversion to float32
- File handle management
- Case-insensitive extension detection

## Using These Feature Files

### With behave

1. Install behave:
   ```bash
   pip install behave
   ```

2. Create step definitions in `features/steps/` directory

3. Run tests:
   ```bash
   behave
   ```

### With pytest-bdd

1. Install pytest-bdd:
   ```bash
   pip install pytest-bdd
   ```

2. Create step definitions in test files

3. Run tests:
   ```bash
   pytest
   ```

### As Documentation

Simply read the feature files to understand the expected behavior of each model class. The Given-When-Then format makes it easy to understand:

- **Given**: Preconditions and setup
- **When**: Action being performed
- **Then**: Expected outcome

## Example Scenario

```gherkin
Scenario: Successfully load a Safetensor model file
  Given a Safetensor model file path "models/test_model.safetensors"
  When I initialize a SafetensorModel with the file path
  Then the model should load successfully
  And the model should contain tensor data
  And tensor names should be cached
```

This scenario describes:
1. Starting condition: A valid safetensor file exists
2. Action: Creating a SafetensorModel instance
3. Expected results: Model loads, contains data, and caches tensor names

## Scenario Outlines

Many scenarios use `Scenario Outline` with `Examples` tables to test multiple cases:

```gherkin
Scenario Outline: Validate tensor types
  Given a loaded SafetensorModel
  And a tensor with dtype "<tensor_dtype>"
  When I call the valid method for the tensor
  Then the method should return "<valid>" for type validation

  Examples:
    | tensor_dtype | valid |
    | float32      | True  |
    | float16      | True  |
    | int8         | True  |
    | int64        | False |
```

This runs the scenario multiple times with different inputs from the Examples table.

## Benefits

1. **Clear Specifications**: Non-technical stakeholders can understand the requirements
2. **Test Coverage**: Comprehensive scenarios ensure thorough testing
3. **Living Documentation**: Features stay up-to-date as code evolves
4. **Regression Prevention**: Scenarios document expected behavior to prevent breaking changes
5. **Consistent Interface**: Protocol features ensure all models behave consistently

## Contributing

When adding new functionality to model classes:

1. Add or update relevant scenarios in the feature files
2. Ensure scenarios follow the Given-When-Then format
3. Use Scenario Outlines for multiple test cases
4. Keep scenarios focused on one behavior at a time
5. Update this README if adding new feature files

## Relationship to Code

The feature files correspond to the following source files:

- `model_protocol.feature` → `gguf_visualizers/model_classes/model_abstract_class.py`
- `gguf_model.feature` → `gguf_visualizers/model_classes/gguf_model.py`
- `torch_model.feature` → `gguf_visualizers/model_classes/torch_model.py`
- `safetensor_model.feature` → `gguf_visualizers/model_classes/safetensor_model.py`
