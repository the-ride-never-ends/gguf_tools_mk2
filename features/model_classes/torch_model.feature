Feature: PyTorch Model Loading
  As a developer using the gguf_tools_mk2 suite
  I want to load and interact with PyTorch model files
  So that I can visualize and analyze tensor data

  Background:
    Given the torch Python module is installed
    And a valid PyTorch model file exists

  Scenario: Successfully load a PyTorch model file
    Given a PyTorch model file path "models/test_model.pth"
    When I initialize a TorchModel with the file path
    Then the model should load successfully
    And the model should contain tensor data

  Scenario: Load PyTorch model without torch module installed
    Given the torch Python module is not installed
    When I attempt to initialize a TorchModel
    Then the system should exit with an error
    And an error message should indicate the torch module is required

  Scenario: Load PyTorch model with CPU mapping
    Given a PyTorch model file
    When I initialize a TorchModel
    Then the model should be loaded with map_location set to "cpu"
    And the model should use memory mapping (mmap=True)

  Scenario: Retrieve tensor names from PyTorch model
    Given a loaded TorchModel
    When I call the tensor_names method
    Then I should receive an iterable of tensor names
    And the tensor names should match the model's state dict keys

  Scenario: Tensors are squeezed on load
    Given a PyTorch model with tensors that have singleton dimensions
    When I initialize a TorchModel
    Then all tensors should be squeezed
    And singleton dimensions should be removed

  Scenario Outline: Validate tensor existence and properties
    Given a loaded TorchModel
    And a tensor named "<tensor_name>"
    When I call the valid method with "<tensor_name>"
    Then the method should return "<valid>" and "<message>"

    Examples:
      | tensor_name           | valid | message           |
      | existing_tensor       | True  | OK                |
      | nonexistent_tensor    | False | Tensor not found  |

  Scenario Outline: Validate tensor types
    Given a loaded TorchModel
    And a tensor with dtype "<tensor_dtype>"
    When I call the valid method for the tensor
    Then the method should return "<valid>" for type validation

    Examples:
      | tensor_dtype | valid |
      | float32      | True  |
      | float16      | True  |
      | bfloat16     | True  |
      | int32        | False |
      | int64        | False |
      | uint8        | False |

  Scenario Outline: Validate tensor dimensions
    Given a loaded TorchModel
    And a tensor with "<dimensions>" dimensions
    When I call the valid method for the tensor
    Then the method should return "<valid>" for dimension validation

    Examples:
      | dimensions | valid |
      | 1          | True  |
      | 2          | True  |
      | 3          | False |
      | 4          | False |

  Scenario Outline: Get tensor data as float32
    Given a loaded TorchModel
    And a tensor with dtype "<tensor_dtype>"
    When I call get_as_f32 for the tensor
    Then the tensor should be converted to float32 dtype
    And I should receive a numpy array with dtype float32
    And the array shape should match the original tensor shape

    Examples:
      | tensor_dtype |
      | float32      |
      | float16      |
      | bfloat16     |

  Scenario: Get float32 tensor as numpy array
    Given a loaded TorchModel
    And a tensor with dtype float32
    When I call get_as_f32 for the tensor
    Then the tensor should be converted to numpy format
    And the dtype should remain float32

  Scenario: Get float16 tensor as numpy array
    Given a loaded TorchModel
    And a tensor with dtype float16
    When I call get_as_f32 for the tensor
    Then the tensor should be converted to float32 dtype
    And then converted to numpy format
    And the result should have dtype float32

  Scenario: Get bfloat16 tensor as numpy array
    Given a loaded TorchModel
    And a tensor with dtype bfloat16
    When I call get_as_f32 for the tensor
    Then the tensor should be converted to float32 dtype
    And then converted to numpy format
    And the result should have dtype float32

  Scenario: Get tensor type name
    Given a loaded TorchModel
    And a tensor with a specific dtype
    When I call get_type_name for the tensor
    Then I should receive the dtype as a string
    And the string should represent the torch dtype

  Scenario: Handle missing tensor in get_as_f32
    Given a loaded TorchModel
    When I call get_as_f32 with a nonexistent tensor name
    Then a KeyError should be raised

  Scenario: Handle missing tensor in get_type_name
    Given a loaded TorchModel
    When I call get_type_name with a nonexistent tensor name
    Then a KeyError should be raised
