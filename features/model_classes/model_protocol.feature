Feature: Model Protocol Interface
  As a developer using the gguf_tools_mk2 suite
  I want all model classes to implement a consistent interface
  So that I can work with different model types interchangeably

  Background:
    Given the Model protocol is defined
    And model classes implement the Model protocol

  Scenario Outline: Model classes implement required methods
    Given a model class "<model_class>"
    Then the class should have an __init__ method accepting filename parameter
    And the class should have a tensor_names method returning an Iterable
    And the class should have a valid method accepting key parameter
    And the class should have a get_as_f32 method accepting key parameter
    And the class should have a get_type_name method accepting key parameter

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: __init__ method signature
    Given a model class "<model_class>"
    Then the __init__ method should accept filename as Path or str
    And the __init__ method should not return a value

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: tensor_names method signature
    Given a model class "<model_class>"
    Then the tensor_names method should take no parameters
    And the tensor_names method should return an Iterable of strings

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: valid method signature
    Given a model class "<model_class>"
    Then the valid method should accept a key parameter as string
    And the valid method should return a tuple of bool and optional string
    And the tuple should indicate validity and error message

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: get_as_f32 method signature
    Given a model class "<model_class>"
    Then the get_as_f32 method should accept a key parameter as string
    And the get_as_f32 method should return a numpy array
    And the numpy array should have dtype float32

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: get_type_name method signature
    Given a model class "<model_class>"
    Then the get_type_name method should accept a key parameter as string
    And the get_type_name method should return a string

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Consistent error handling for missing dependencies
    Given a model class "<model_class>"
    And the required module "<required_module>" is not installed
    When I attempt to initialize the model
    Then the system should exit with an error code
    And an error message should be logged

    Examples:
      | model_class       | required_module |
      | GGUFModel         | gguf            |
      | TorchModel        | torch           |
      | SafetensorModel   | safetensors     |

  Scenario Outline: Consistent validation behavior
    Given any model class implementing Model protocol
    When I validate a nonexistent tensor
    Then the valid method should return False
    And the message should be "Tensor not found"

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Consistent dimension validation
    Given any model class implementing Model protocol
    And a tensor with more than 2 dimensions
    When I validate the tensor
    Then the valid method should return False
    And the message should be "Unhandled dimensions"

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario: Polymorphic usage of model classes
    Given a function that accepts a Model protocol instance
    When I pass a GGUFModel instance
    Then the function should work correctly
    When I pass a TorchModel instance
    Then the function should work correctly
    When I pass a SafetensorModel instance
    Then the function should work correctly

  Scenario: Type hints compatibility
    Given the Model protocol with type hints
    Then GGUFModel should satisfy the protocol
    And TorchModel should satisfy the protocol
    And SafetensorModel should satisfy the protocol
    And type checkers should not report errors
