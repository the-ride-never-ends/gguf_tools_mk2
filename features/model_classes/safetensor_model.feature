Feature: Safetensor Model Loading
  As a developer using the gguf_tools_mk2 suite
  I want to load and interact with Safetensor model files
  So that I can visualize and analyze tensor data

  Background:
    Given the safetensors Python module is installed
    And a valid Safetensor model file exists

  Scenario: Successfully load a Safetensor model file
    Given a Safetensor model file path "models/test_model.safetensors"
    When I initialize a SafetensorModel with the file path
    Then the model should load successfully
    And the model should contain tensor data
    And tensor names should be cached

  Scenario: Load Safetensor model without safetensors module installed
    Given the safetensors Python module is not installed
    When I attempt to initialize a SafetensorModel
    Then the system should exit with an error
    And an error message should indicate the safetensors module is required

  Scenario: Load Safetensor model with metadata
    Given a Safetensor model file with metadata
    When I initialize a SafetensorModel
    Then the model should load successfully
    And the metadata should be cached
    And tensor names should be extracted from the file

  Scenario: Lazy loading of tensors
    Given a Safetensor model file
    When I initialize a SafetensorModel
    Then only the tensor keys should be loaded initially
    And tensor data should not be loaded until requested

  Scenario: Retrieve tensor names from Safetensor model
    Given a loaded SafetensorModel
    When I call the tensor_names method
    Then I should receive an iterable of tensor names
    And the tensor names should match the safetensor file's keys

  Scenario Outline: Validate tensor existence and properties
    Given a loaded SafetensorModel
    And a tensor named "<tensor_name>"
    When I call the valid method with "<tensor_name>"
    Then the method should return "<valid>" and "<message>"

    Examples:
      | tensor_name           | valid | message           |
      | existing_tensor       | True  | OK                |
      | nonexistent_tensor    | False | Tensor not found  |

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
      | int16        | True  |
      | int32        | True  |
      | int64        | False |
      | uint8        | False |

  Scenario Outline: Validate tensor dimensions
    Given a loaded SafetensorModel
    And a tensor with "<dimensions>" dimensions
    When I call the valid method for the tensor
    Then the method should return "<valid>" for dimension validation

    Examples:
      | dimensions | valid |
      | 1          | True  |
      | 2          | True  |
      | 3          | False |
      | 4          | False |

  Scenario: Validation opens and closes file properly
    Given a loaded SafetensorModel
    When I call the valid method for a tensor
    Then the file should be opened with framework "numpy" and device "cpu"
    And the file should be closed after validation
    And no file handles should remain open

  Scenario Outline: Get tensor data as float32
    Given a loaded SafetensorModel
    And a tensor with dtype "<tensor_dtype>"
    When I call get_as_f32 for the tensor
    Then I should receive a numpy array with dtype float32
    And the array shape should match the original tensor shape

    Examples:
      | tensor_dtype |
      | float32      |
      | float16      |
      | int8         |
      | int16        |
      | int32        |

  Scenario: Get float32 tensor without conversion
    Given a loaded SafetensorModel
    And a tensor with dtype float32
    When I call get_as_f32 for the tensor
    Then the original tensor data should be returned
    And no type conversion should occur

  Scenario: Get non-float32 tensor with conversion
    Given a loaded SafetensorModel
    And a tensor with dtype float16
    When I call get_as_f32 for the tensor
    Then the tensor should be converted to float32 using astype
    And the result should have dtype float32

  Scenario: Get tensor as float32 opens and closes file
    Given a loaded SafetensorModel
    When I call get_as_f32 for a tensor
    Then the file should be opened with framework "numpy" and device "cpu"
    And the tensor should be retrieved using get_tensor
    And the file should be closed after retrieval
    And no file handles should remain open

  Scenario: Get tensor type name
    Given a loaded SafetensorModel
    And a tensor with a specific dtype
    When I call get_type_name for the tensor
    Then I should receive the dtype as a string
    And the string should represent the numpy dtype

  Scenario: Get type name opens and closes file
    Given a loaded SafetensorModel
    When I call get_type_name for a tensor
    Then the file should be opened with framework "numpy" and device "cpu"
    And the tensor should be retrieved using get_tensor
    And the file should be closed after retrieval
    And no file handles should remain open

  Scenario: Memory efficiency with large models
    Given a large Safetensor model file
    When I initialize a SafetensorModel
    Then only metadata and tensor keys should be loaded
    And the memory footprint should be minimal
    And tensors should be loaded on-demand

  Scenario: Case-insensitive file extension detection
    Given a Safetensor file with extension ".SAFETENSORS"
    When the file is detected by the system
    Then it should be recognized as a Safetensor model
    And SafetensorModel should be used to load it

  Scenario: Handle concurrent tensor access
    Given a loaded SafetensorModel
    When I access multiple tensors sequentially
    Then each access should open and close the file
    And there should be no file handle conflicts
    And all tensor data should be retrieved correctly
