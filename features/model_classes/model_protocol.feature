Feature: Model Protocol Interface
  As a developer using the gguf_tools_mk2 suite
  I want all model classes to implement a consistent interface
  So that I can work with different model types interchangeably

  Background:
    Given the Model protocol is defined
    Given model classes implement the Model protocol

  Scenario Outline: Model class has __init__ method
    Given a model class "<model_class>"
    Then the class should have an __init__ method accepting filename parameter

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Model class has tensor_names method
    Given a model class "<model_class>"
    Then the class should have a tensor_names method returning an Iterable

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Model class has valid method
    Given a model class "<model_class>"
    Then the class should have a valid method accepting key parameter

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Model class has get_as_f32 method
    Given a model class "<model_class>"
    Then the class should have a get_as_f32 method accepting key parameter

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Model class has get_type_name method
    Given a model class "<model_class>"
    Then the class should have a get_type_name method accepting key parameter

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: __init__ method accepts filename parameter
    Given a model class "<model_class>"
    Then the __init__ method should accept filename as Path or str

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: __init__ method returns no value
    Given a model class "<model_class>"
    Then the __init__ method should not return a value

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: tensor_names method takes no parameters
    Given a model class "<model_class>"
    Then the tensor_names method should take no parameters

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: tensor_names method returns Iterable
    Given a model class "<model_class>"
    Then the tensor_names method should return an Iterable of strings

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: valid method accepts key parameter
    Given a model class "<model_class>"
    Then the valid method should accept a key parameter as string

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: valid method returns tuple
    Given a model class "<model_class>"
    Then the valid method should return a tuple of bool and optional string

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: valid method tuple indicates validity
    Given a model class "<model_class>"
    Then the tuple should indicate validity and error message

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: get_as_f32 method accepts key parameter
    Given a model class "<model_class>"
    Then the get_as_f32 method should accept a key parameter as string

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: get_as_f32 method returns numpy array
    Given a model class "<model_class>"
    Then the get_as_f32 method should return a numpy array

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: get_as_f32 returns float32 dtype
    Given a model class "<model_class>"
    Then the numpy array should have dtype float32

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: get_type_name method accepts key parameter
    Given a model class "<model_class>"
    Then the get_type_name method should accept a key parameter as string

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: get_type_name method returns string
    Given a model class "<model_class>"
    Then the get_type_name method should return a string

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: System exits when dependency missing
    Given a model class "<model_class>"
    Given the required module "<required_module>" is not installed
    When I attempt to initialize the model
    Then the system should exit with an error code

    Examples:
      | model_class       | required_module |
      | GGUFModel         | gguf            |
      | TorchModel        | torch           |
      | SafetensorModel   | safetensors     |

  Scenario Outline: Error message logged for missing dependency
    Given a model class "<model_class>"
    Given the required module "<required_module>" is not installed
    When I attempt to initialize the model
    Then an error message should be logged

    Examples:
      | model_class       | required_module |
      | GGUFModel         | gguf            |
      | TorchModel        | torch           |
      | SafetensorModel   | safetensors     |

  Scenario Outline: Valid method returns False for nonexistent tensor
    Given any model class implementing Model protocol
    When I validate a nonexistent tensor
    Then the valid method should return False

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Message for nonexistent tensor
    Given any model class implementing Model protocol
    When I validate a nonexistent tensor
    Then the message should be "Tensor not found"

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Valid method returns False for high-dimensional tensors
    Given any model class implementing Model protocol
    Given a tensor with more than 2 dimensions
    When I validate the tensor
    Then the valid method should return False

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario Outline: Message for high-dimensional tensors
    Given any model class implementing Model protocol
    Given a tensor with more than 2 dimensions
    When I validate the tensor
    Then the message should be "Unhandled dimensions"

    Examples:
      | model_class       |
      | GGUFModel         |
      | TorchModel        |
      | SafetensorModel   |

  Scenario: GGUFModel works with polymorphic function
    Given a function that accepts a Model protocol instance
    When I pass a GGUFModel instance
    Then the function should work correctly

  Scenario: TorchModel works with polymorphic function
    Given a function that accepts a Model protocol instance
    When I pass a TorchModel instance
    Then the function should work correctly

  Scenario: SafetensorModel works with polymorphic function
    Given a function that accepts a Model protocol instance
    When I pass a SafetensorModel instance
    Then the function should work correctly

  Scenario: GGUFModel satisfies protocol
    Given the Model protocol with type hints
    Then GGUFModel should satisfy the protocol

  Scenario: TorchModel satisfies protocol
    Given the Model protocol with type hints
    Then TorchModel should satisfy the protocol

  Scenario: SafetensorModel satisfies protocol
    Given the Model protocol with type hints
    Then SafetensorModel should satisfy the protocol

  Scenario: Type checkers report no errors
    Given the Model protocol with type hints
    Then type checkers should not report errors
