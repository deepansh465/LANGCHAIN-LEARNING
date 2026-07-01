# Import BaseModel to create a data model with validation
# EmailStr validates that the value is a proper email address
# Field lets us add constraints and metadata to fields
from pydantic import BaseModel, EmailStr, Field

# Optional means the value can either be of the given type or None
from typing import Optional


# Define a Student model
# Any data passed to this model will be validated automatically
class Student(BaseModel):

    # Default value is "nitish"
    # If no name is provided, this value will be used
    name: str = 'nitish'

    # Age can be an integer or None
    # Default value is None
    age: Optional[int] = None

    # Must contain a valid email address
    email: EmailStr

    # CGPA must be:
    # greater than 0 (gt=0)
    # less than 10 (lt=10)
    # default value = 5
    # description is useful when generating documentation
    cgpa: float = Field(
        gt=0,
        lt=10,
        default=5,
        description='A decimal value representing the cgpa of the student'
    )


# Raw data received from user/API/database
new_student = {
    'age': '32',                # String value
    'email': 'abc@gmail.com'
}

# Create Student object
# ** unpacks dictionary into keyword arguments
# Pydantic automatically converts '32' (string) -> 32 (int)
student = Student(**new_student)

# Convert model object into a Python dictionary
student_dict = dict(student)

# Access and print age from dictionary
print(student_dict['age'])

# Convert model into JSON string
# Useful when sending data through APIs
student_json = student.model_dump_json()

# Print JSON representation
print(student_json)