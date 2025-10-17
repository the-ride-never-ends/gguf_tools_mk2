Feature: Safetensor Model Loading
  As a developer using the gguf_tools_mk2 suite
  I want to load and interact with Safetensor model files
  So that I can visualize and analyze tensor data

  Background:
    Given the safetensors Python module is installed
    Given a valid Safetensor model file exists

  Scenario: Successfully load a Safetensor model file
    Given a Safetensor model file path "models/test_model.safetensors"
    When I initialize a SafetensorModel with the file path
    Then the model should load successfully

  Scenario: Loaded Safetensor model contains tensor data
    Given a Safetensor model file path "models/test_model.safetensors"
    When I initialize a SafetensorModel with the file path
    Then the model should contain tensor data

  Scenario: Tensor names are cached on load
    Given a Safetensor model file path "models/test_model.safetensors"
    When I initialize a SafetensorModel with the file path
    Then tensor names should be cached

  Scenario: Load Safetensor model without safetensors module installed
    Given the safetensors Python module is not installed
    When I attempt to initialize a SafetensorModel
    Then the system should exit with an error

  Scenario: Error message when safetensors module is missing
    Given the safetensors Python module is not installed
    When I attempt to initialize a SafetensorModel
    Then an error message should indicate the safetensors module is required

  Scenario: Load Safetensor model with metadata
    Given a Safetensor model file with metadata
    When I initialize a SafetensorModel
    Then the model should load successfully

  Scenario: Metadata is cached on load
    Given a Safetensor model file with metadata
    When I initialize a SafetensorModel
    Then the metadata should be cached

  Scenario: Tensor names are extracted from file
    Given a Safetensor model file with metadata
    When I initialize a SafetensorModel
    Then tensor names should be extracted from the file

  Scenario: Lazy loading of tensors
    Given a Safetensor model file
    When I initialize a SafetensorModel
    Then only the tensor keys should be loaded initially

  Scenario: Tensor data not loaded until requested
    Given a Safetensor model file
    When I initialize a SafetensorModel
    Then tensor data should not be loaded until requested

  Scenario: Retrieve tensor names from Safetensor model
    Given a loaded SafetensorModel
    When I call the tensor_names method
    Then I should receive an iterable of tensor names

  Scenario: Tensor names match safetensor file keys
    Given a loaded SafetensorModel
    When I call the tensor_names method
    Then the tensor names should match the safetensor file's keys

  Scenario Outline: Validate tensor existence and properties
    Given a loaded SafetensorModel
    Given a tensor named "<tensor_name>"
    When I call the valid method with "<tensor_name>"
    Then the method should return "<valid>"
    Then the method should return message "<message>"

    Examples:
      | tensor_name           | valid | message           |
      | existing_tensor       | True  | OK                |
      | nonexistent_tensor    | False | Tensor not found  |

  Scenario Outline: Validate tensor types
    Given a loaded SafetensorModel
    Given a tensor with dtype "<tensor_dtype>"
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
    Given a tensor with "<dimensions>" dimensions
    When I call the valid method for the tensor
    Then the method should return "<valid>" for dimension validation

    Examples:
      | dimensions | valid |
      | 1          | True  |
      | 2          | True  |
      | 3          | False |
      | 4          | False |

  Scenario: Validation opens file with correct parameters
    Given a loaded SafetensorModel
    When I call the valid method for a tensor
    Then the file should be opened with framework "numpy" and device "cpu"

  Scenario: File is closed after validation
    Given a loaded SafetensorModel
    When I call the valid method for a tensor
    Then the file should be closed after validation

  Scenario: No file handles remain open after validation
    Given a loaded SafetensorModel
    When I call the valid method for a tensor
    Then no file handles should remain open

  Scenario Outline: Get tensor data as float32
    Given a loaded SafetensorModel
    Given a tensor with dtype "<tensor_dtype>"
    When I call get_as_f32 for the tensor
    Then I should receive a numpy array with dtype float32

  Scenario Outline: Safetensor array shape matches original
    Given a loaded SafetensorModel
    Given a tensor with dtype "<tensor_dtype>"
    When I call get_as_f32 for the tensor
    Then the array shape should match the original tensor shape

    Examples:
      | tensor_dtype |
      | float32      |
      | float16      |
      | int8         |
      | int16        |
      | int32        |

  Scenario: Get float32 tensor without conversion
    Given a loaded SafetensorModel
    Given a tensor with dtype float32
    When I call get_as_f32 for the tensor
    Then the original tensor data should be returned

  Scenario: No type conversion for float32 tensors
    Given a loaded SafetensorModel
    Given a tensor with dtype float32
    When I call get_as_f32 for the tensor
    Then no type conversion should occur

  Scenario: Get non-float32 tensor with conversion
    Given a loaded SafetensorModel
    Given a tensor with dtype float16
    When I call get_as_f32 for the tensor
    Then the tensor should be converted to float32 using astype

  Scenario: Non-float32 result has dtype float32
    Given a loaded SafetensorModel
    Given a tensor with dtype float16
    When I call get_as_f32 for the tensor
    Then the result should have dtype float32

  Scenario: Get tensor as float32 opens file correctly
    Given a loaded SafetensorModel
    When I call get_as_f32 for a tensor
    Then the file should be opened with framework "numpy" and device "cpu"

  Scenario: Tensor is retrieved using get_tensor method
    Given a loaded SafetensorModel
    When I call get_as_f32 for a tensor
    Then the tensor should be retrieved using get_tensor

  Scenario: File is closed after tensor retrieval
    Given a loaded SafetensorModel
    When I call get_as_f32 for a tensor
    Then the file should be closed after retrieval

  Scenario: No file handles remain after get_as_f32
    Given a loaded SafetensorModel
    When I call get_as_f32 for a tensor
    Then no file handles should remain open

  Scenario: Get tensor type name
    Given a loaded SafetensorModel
    Given a tensor with a specific dtype
    When I call get_type_name for the tensor
    Then I should receive the dtype as a string

  Scenario: Type name represents numpy dtype
    Given a loaded SafetensorModel
    Given a tensor with a specific dtype
    When I call get_type_name for the tensor
    Then the string should represent the numpy dtype

  Scenario: Get type name opens file correctly
    Given a loaded SafetensorModel
    When I call get_type_name for a tensor
    Then the file should be opened with framework "numpy" and device "cpu"

  Scenario: Tensor retrieved when getting type name
    Given a loaded SafetensorModel
    When I call get_type_name for a tensor
    Then the tensor should be retrieved using get_tensor

  Scenario: File closed after getting type name
    Given a loaded SafetensorModel
    When I call get_type_name for a tensor
    Then the file should be closed after retrieval

  Scenario: No file handles after get_type_name
    Given a loaded SafetensorModel
    When I call get_type_name for a tensor
    Then no file handles should remain open

  Scenario: Memory efficiency with large models
    Given a large Safetensor model file
    When I initialize a SafetensorModel
    Then only metadata and tensor keys should be loaded

  Scenario: Minimal memory footprint on load
    Given a large Safetensor model file
    When I initialize a SafetensorModel
    Then the memory footprint should be minimal

  Scenario: Tensors loaded on-demand
    Given a large Safetensor model file
    When I initialize a SafetensorModel
    Then tensors should be loaded on-demand

  Scenario: Case-insensitive file extension detection
    Given a Safetensor file with extension ".SAFETENSORS"
    When the file is detected by the system
    Then it should be recognized as a Safetensor model

  Scenario: SafetensorModel used for uppercase extension
    Given a Safetensor file with extension ".SAFETENSORS"
    When the file is detected by the system
    Then SafetensorModel should be used to load it

  Scenario: Handle concurrent tensor access
    Given a loaded SafetensorModel
    When I access multiple tensors sequentially
    Then each access should open and close the file

  Scenario: No file handle conflicts with sequential access
    Given a loaded SafetensorModel
    When I access multiple tensors sequentially
    Then there should be no file handle conflicts

  Scenario: All tensor data retrieved correctly
    Given a loaded SafetensorModel
    When I access multiple tensors sequentially
    Then all tensor data should be retrieved correctly
