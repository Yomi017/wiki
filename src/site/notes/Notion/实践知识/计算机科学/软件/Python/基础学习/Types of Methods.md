---
{"dg-publish":true,"permalink":"/notion////python//types-of-methods/"}
---

## Comparison of Python Method Types

|   |   |   |   |
|---|---|---|---|
|Feature|Instance Method|Class Method (`@classmethod`)|Static Method (`@staticmethod`)|
|**Decorator**|None|`@classmethod`|`@staticmethod`|
|**First Argument**|`self` (the instance object)|`cls` (the class itself)|None (no implicit first argument)|
|**Access Instance Data**|Yes (via `self.attribute`)|No (not directly; must create an instance via `cls()`)|No (unless an instance is passed as an argument)|
|**Access Class Data**|Yes (via `self.__class__.attribute` or `ClassName.attribute`)|Yes (via `cls.attribute`)|Yes (via `ClassName.attribute`)|
|**How It's Called**|On an instance: `obj.method()`|On class: `ClassName.method()` <br> On instance: `obj.method()` (but `cls` is still the class)|On class: `ClassName.method()` <br> On instance: `obj.method()`|
|**Primary Purpose**|Operate on instance-specific data/state.|Factory methods, operate on class-level data/state.|Utility functions logically grouped with the class.|

---

## Detailed Explanations and Examples

### 1. Instance Method

These are the most common type of methods. They operate on an instance of the class and have access to the instance's attributes and methods via the `self` argument.

**Example Code:**

```Python
class Dog:
    species = "Canis familiaris"  # Class attribute

    def __init__(self, name, age):
        self.name = name  # Instance attribute
        self.age = age    # Instance attribute

    # Instance method
    def describe(self):
        return f"{self.name} is {self.age} years old."

    # Instance method that can access class attributes
    def get_species(self):
        return f"{self.name} belongs to the species: {self.__class__.species}" # or Dog.species

# Usage
my_dog = Dog("Buddy", 3)
print(my_dog.describe())        # Output: Buddy is 3 years old.
print(my_dog.get_species())     # Output: Buddy belongs to the species: Canis familiaris
```

**Benefits:**

- **Operates on Instance State:** Can directly access and modify the data specific to each object (instance variables).
    - 操作实例状态：可以直接访问和修改特定于每个对象的数据（实例变量）。
- **Object-Oriented Core:** Fundamental to object-oriented programming, allowing objects to have behaviors tied to their data.
    - 面向对象核心：面向对象编程的基础，允许对象拥有与其数据关联的行为。

**Use Cases:**

- Modifying or accessing instance-specific attributes (e.g., `user.set_email()`, `order.calculate_total()`).
    - 修改或访问特定于实例的属性（例如，`user.set_email()`，`order.calculate_total()`）。
- Performing actions related to a specific object (e.g., `file.read()`, `car.start_engine()`).
    - 执行与特定对象相关的操作（例如，`file.read()`，`car.start_engine()`）。
- Implementing most of the object's behavior.
    - 实现对象的大部分行为。

---

### 2. Class Method (`@classmethod`)

Class methods are bound to the class and not the instance of the class. They receive the class itself as the first argument, conventionally named `cls`. They can modify class state that applies across all instances of the class.

**Example Code:**

```Python
class Car:
    num_cars_created = 0  # Class attribute

    def __init__(self, color, model):
        self.color = color
        self.model = model
        Car.num_cars_created += 1

    @classmethod
    def get_num_cars(cls):
        # cls refers to the Car class
        return f"Number of cars created: {cls.num_cars_created}"

    @classmethod
    def create_blue_sedan(cls, model_name):
        # cls() is equivalent to Car() here.
        # This is a factory method.
        print(f"Factory: Creating a blue sedan of type {cls.__name__} with model {model_name}")
        return cls(color="blue", model=model_name)

class ElectricCar(Car):
    # This subclass will correctly use ElectricCar as cls in inherited classmethods
    pass

# Usage
car1 = Car("red", "SUV")
car2 = Car("black", "Hatchback")

print(Car.get_num_cars())         # Output: Number of cars created: 2
print(car1.get_num_cars())        # Output: Number of cars created: 2 (can be called on instance too)

blue_swift = Car.create_blue_sedan("Swift")
print(f"Created: {blue_swift.color} {blue_swift.model}") # Output: Created: blue Swift

# Demonstrating inheritance with class methods
electric_tesla = ElectricCar.create_blue_sedan("Model S") # cls will be ElectricCar
print(f"Created Electric: {electric_tesla.color} {electric_tesla.model}")
print(Car.get_num_cars()) # Will be 4 now (car1, car2, blue_swift, electric_tesla)
```

**Benefits:**

- **Access to Class (**`**cls**`**):** The method knows its class. This is especially useful in inheritance, as `cls` will be the subclass if called from a subclass.
    - 访问类本身 (`cls`)：方法知道它属于哪个类。这在继承中尤其有用，因为如果从子类调用，`cls` 将是子类。
- **Factory Pattern Implementation:** Ideal for creating factory methods that can instantiate the class (or subclasses) in various ways (e.g., `Date.from_timestamp()`, `User.from_json()`).
    - 工厂模式实现：非常适合创建工厂方法，这些方法可以以各种方式实例化类（或子类）（例如，`Date.from_timestamp()`，`User.from_json()`）。
- **Operate on Class Attributes:** Can directly read or modify class-level attributes shared by all instances.
    - 操作类属性：可以直接读取或修改所有实例共享的类级别属性。
- **No Instance Needed:** Can be called on the class itself without needing to create an instance first.
    - 不需要实例：可以在类本身上调用，而无需先创建实例。

**Use Cases:**

- **Factory Methods / Alternative Constructors:** Providing different ways to create instances (e.g., `datetime.datetime.now()`, `dict.fromkeys()`).
    - 工厂方法/备选构造函数：提供创建实例的不同方式（例如，`datetime.datetime.now()`，`dict.fromkeys()`）。
- **Managing Class-Level State:** Tracking the number of instances created, managing a registry of subclasses, or setting class-wide configurations.
    - 管理类级别状态：跟踪已创建的实例数量，管理子类的注册表，或设置类范围的配置。
- **Inheritance-Aware Methods:** When you want a method in a base class to work correctly with subclasses (e.g., a `create_default()` method in a base class should create an instance of the subclass it's called on).
    - 继承感知方法：当你希望基类中的方法能够正确地与子类一起工作时（例如，基类中的 `create_default()` 方法应该创建调用它的子类的实例）。

---

### 3. Static Method (`@staticmethod`)

Static methods do not receive an implicit first argument (`self` or `cls`). They are essentially regular functions that live inside a class's namespace, primarily because they logically belong to the class, but they don't operate on instance or class state.

**Example Code:**

```Python
class MathUtils:
    PI = 3.14159 # Class attribute, can be accessed via MathUtils.PI

    @staticmethod
    def add(x, y):
        # This method doesn't use self or cls
        return x + y

    @staticmethod
    def is_even(num):
        return num % 2 == 0

    @staticmethod
    def circle_area(radius):
        # Can access class attributes via class name if needed
        return MathUtils.PI * radius * radius

# Usage
sum_val = MathUtils.add(5, 3)
print(f"Sum: {sum_val}")  # Output: Sum: 8

print(f"Is 4 even? {MathUtils.is_even(4)}")  # Output: Is 4 even? True
print(f"Area of circle with radius 2: {MathUtils.circle_area(2)}") # Output: Area of circle with radius 2: 12.56636

# Can also be called on an instance, but it's not common practice as it's misleading
# utils_instance = MathUtils()
# print(utils_instance.add(10, 2)) # Works, but add() doesn't use the instance
```

**Benefits:**

- **Logical Grouping:** Organizes utility functions that are conceptually related to the class, improving code structure and discoverability.
    - 逻辑分组：组织与类在概念上相关的实用函数，提高代码结构和可发现性。
- **No Implicit Arguments:** The method signature is cleaner as it doesn't automatically receive `self` or `cls`. This clearly indicates that the method does not depend on instance or class state.
    - 无隐式参数：方法签名更清晰，因为它不会自动接收 `self` 或 `cls`。这清楚地表明该方法不依赖于实例或类状态。
- **Namespace Organization:** Keeps utility functions out of the global namespace, preventing potential naming conflicts.
    - 命名空间组织：将实用函数保留在类的命名空间内，避免潜在的命名冲突。
- **Clear Intent:** Signals to other developers that this method is independent and should not modify instance or class state.
    - 清晰意图：向其他开发人员表明此方法是独立的，不应修改实例或类状态。

**Use Cases:**

- **Utility/Helper Functions:** Functions that perform a task related to the class but don't need access to instance or class-specific data (e.g., validation functions, conversion utilities).
    - 实用/辅助函数：执行与类相关的任务但不需要访问实例或类特定数据的函数（例如，验证函数，转换实用程序）。
- Grouping related functions within a class's namespace for better organization (e.g., a `StringUtils` class with various static methods for string manipulation).
    - 在类的命名空间内对相关函数进行分组，以实现更好的组织（例如，包含各种字符串操作静态方法的 `StringUtils` 类）。
- When a piece of functionality is related to a class, but doesn't require an instance or the class itself to operate.
    - 当某项功能与类相关，但其操作不需要实例或类本身时。