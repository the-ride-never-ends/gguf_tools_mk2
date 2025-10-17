Feature: GGUF Model Loading
  As a developer using the gguf_tools_mk2 suite
  I want to load and interact with GGUF model files
  So that I can visualize and analyze tensor data

  Background:
    Given the gguf Python module is installed
    Given a valid GGUF model file exists

  Scenario: Successfully load a GGUF model file
    Given a GGUF model file path "models/test_model.gguf"
    When I initialize a GGUFModel with the file path
    Then the model should load successfully

  Scenario: Loaded GGUF model contains tensor data
    Given a GGUF model file path "models/test_model.gguf"
    When I initialize a GGUFModel with the file path
    Then the model should contain tensor data

  Scenario: Load GGUF model without gguf module installed
    Given the gguf Python module is not installed
    When I attempt to initialize a GGUFModel
    Then the system should exit with an error

  Scenario: Error message when gguf module is missing
    Given the gguf Python module is not installed
    When I attempt to initialize a GGUFModel
    Then an error message should indicate the gguf module is required

  Scenario: Retrieve tensor names from GGUF model
    Given a loaded GGUFModel
    When I call the tensor_names method
    Then I should receive an iterable of tensor names

  Scenario: Tensor names match model content
    Given a loaded GGUFModel
    When I call the tensor_names method
    Then the tensor names should match the model's content

  Scenario Outline: Validate tensor existence returns correct status
    Given a loaded GGUFModel with a tensor named "<tensor_name>"
    When I call the valid method with "<tensor_name>"
    Then the method should return "<valid>"

    Examples:
      | tensor_name           | valid |
      | existing_tensor       | True  |
      | nonexistent_tensor    | False |

  Scenario Outline: Validate tensor existence returns correct message
    Given a loaded GGUFModel with a tensor named "<tensor_name>"
    When I call the valid method with "<tensor_name>"
    Then the method should return message "<message>"

    Examples:
      | tensor_name           | message           |
      | existing_tensor       | OK                |
      | nonexistent_tensor    | Tensor not found  |

  Scenario Outline: Validate tensor types
    Given a loaded GGUFModel with a tensor of type "<tensor_type>"
    When I call the valid method for the tensor
    Then the method should return "<valid>" for type validation

    Examples:
      | tensor_type | valid |
      | F16         | True  |
      | F32         | True  |
      | Q8_0        | True  |
      | Q4_0        | False |
      | I8          | False |

  Scenario Outline: Validate tensor dimensions
    Given a loaded GGUFModel with a tensor of "<dimensions>" dimensions
    When I call the valid method for the tensor
    Then the method should return "<valid>" for dimension validation

    Examples:
      | dimensions | valid |
      | 1          | True  |
      | 2          | True  |
      | 3          | False |
      | 4          | False |

  Scenario Outline: Get tensor data as float32
    Given a loaded GGUFModel with a tensor of type "<tensor_type>"
    When I call get_as_f32 for the tensor
    Then I should receive a numpy array with dtype float32

    Examples:
      | tensor_type |
      | F16         |
      | F32         |
      | Q8_0        |

  Scenario Outline: Array shape matches original tensor shape
    Given a loaded GGUFModel with a tensor of type "<tensor_type>"
    When I call get_as_f32 for the tensor
    Then the array shape should match the original tensor shape

    Examples:
      | tensor_type |
      | F16         |
      | F32         |
      | Q8_0        |

  Scenario: Get F16 tensor as float32
    Given a loaded GGUFModel with a tensor of type F16
    When I call get_as_f32 for the tensor
    Then the tensor data should be viewed as float32 dtype

  Scenario: Get F32 tensor as float32
    Given a loaded GGUFModel with a tensor of type F32
    When I call get_as_f32 for the tensor
    Then the original tensor data should be returned

  Scenario: Get Q8_0 tensor as float32
    Given a loaded GGUFModel with a tensor of type Q8_0
    When I call get_as_f32 for the tensor
    Then the tensor should be dequantized

  Scenario: Q8_0 result is reshaped to original tensor shape
    Given a loaded GGUFModel with a tensor of type Q8_0
    When I call get_as_f32 for the tensor
    Then the result should be reshaped to the original tensor shape

  Scenario: Q8_0 result has dtype float32
    Given a loaded GGUFModel with a tensor of type Q8_0
    When I call get_as_f32 for the tensor
    Then the result should have dtype float32

  Scenario: Get unsupported tensor type as float32
    Given a loaded GGUFModel with an unsupported tensor type
    When I call get_as_f32 for the tensor
    Then a ValueError should be raised

  Scenario: Unsupported tensor type error message
    Given a loaded GGUFModel with an unsupported tensor type
    When I call get_as_f32 for the tensor
    Then the error message should indicate "Unhandled tensor type"

  Scenario: Get tensor type name
    Given a loaded GGUFModel with a specific quantization type tensor
    When I call get_type_name for the tensor
    Then I should receive the tensor type name as a string

  Scenario: Tensor type name matches GGMLQuantizationType
    Given a loaded GGUFModel with a specific quantization type tensor
    When I call get_type_name for the tensor
    Then the name should match the GGMLQuantizationType name

  Scenario: Handle missing tensor in get_as_f32
    Given a loaded GGUFModel
    When I call get_as_f32 with a nonexistent tensor name
    Then a KeyError should be raised

  Scenario: Handle missing tensor in get_type_name
    Given a loaded GGUFModel
    When I call get_type_name with a nonexistent tensor name
    Then a KeyError should be raised
