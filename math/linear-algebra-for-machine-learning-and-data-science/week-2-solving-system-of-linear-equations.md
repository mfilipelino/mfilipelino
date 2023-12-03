# Week 2: Solving system of linear equations

Certainly! Solving a system of linear equations can be approached in several ways. Each method has its own advantages and can be more suitable depending on the nature of the equations and the desired solution. I'll explain each of these methods with examples, using markdown format for clarity.

#### Graphical Method

The graphical method involves plotting each equation on a graph and finding the point(s) where they intersect. This method is best suited for systems with two variables.

**Example:**

Consider the system of equations:

1. ( y = 2x + 3 )
2. ( y = -x + 1 )

**Python Code Example:**

```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(-10, 10, 400)
y1 = 2*x + 3
y2 = -x + 1

plt.figure(figsize=(8,6))
plt.plot(x, y1, label='y = 2x + 3')
plt.plot(x, y2, label='y = -x + 1')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.axhline(0, color='black',linewidth=0.5)
plt.axvline(0, color='black',linewidth=0.5)
plt.grid(color = 'gray', linestyle = '--', linewidth = 0.5)
plt.title('Graphical Method')
plt.show()
```

#### Substitution Method

The substitution method involves solving one of the equations for one variable and then substituting this expression into the other equation(s).

**Example:**

Given the system:

1. ( x + y = 10 )
2. ( 2x - y = 0 )

Solve the second equation for ( y ): ( y = 2x )

Substitute ( y ) in the first equation: ( x + 2x = 10 ) ( 3x = 10 ) ( x = \frac{10}{3} )

Then find ( y ): ( y = 2 \times \frac{10}{3} = \frac{20}{3} )

#### Elimination Method

In the elimination method, you manipulate the equations to eliminate one of the variables, making it easier to solve for the other.

**Example:**

Given the system:

1. ( 3x + 2y = 16 )
2. ( 2x - 3y = -11 )

Multiply the first equation by 3 and the second by 2:

1. ( 9x + 6y = 48 )
2. ( 4x - 6y = -22 )

Add the equations to eliminate ( y ): ( 13x = 26 ) ( x = 2 )

Substitute ( x ) back into one of the original equations to find ( y ).

#### Row Echelon Form

The row echelon form (REF) method involves using elementary row operations to transform the matrix of coefficients into an upper triangular form, making it easier to solve.

**Example:**

Consider the system represented in matrix form: ( \begin{bmatrix} 1 & 1 & 10 \ 2 & -1 & 0 \end{bmatrix} )

**Python Code Example to find REF:**

```python
from sympy import Matrix

A = Matrix([[1, 1, 10], [2, -1, 0]])
A_rref = A.rref()

print(A_rref)
```

This will transform the matrix into its row echelon form, from which the solution can be directly read or easily found.

Certainly! Let's go through the Row Echelon Form (REF) process step-by-step using mathematical expressions. To illustrate this, we'll use a simple system of linear equations and convert its coefficient matrix into REF.

#### Example System of Equations:

1. ( 2x + 3y = 8 )
2. ( 4x - y = -2 )

#### Step 1: Write the Augmented Matrix

First, we write the augmented matrix of this system. The augmented matrix includes the coefficients of the variables and the constants from the right-hand side of the equations.

\[ \begin{bmatrix} 2 & 3 & | & 8 \ 4 & -1 & | & -2 \end{bmatrix} ]

#### Step 2: Make the Leading Coefficient of the First Row 1 (if it's not already)

The leading coefficient of the first row is already 1, so we can skip this step. If it wasn't, we would multiply the entire row by the reciprocal of the leading coefficient.

#### Step 3: Zero Out the Elements Below the Leading Coefficient of the First Row

We want to make the first element of the second row zero. To do this, we can replace the second row with the result of ( \text{Row 2} - 2 \times \text{Row 1} ).

Original Row 2: ( \[4 \quad -1 \quad | \quad -2] )

( 2 \times ) Row 1: ( \[4 \quad 6 \quad | \quad 16] )

Subtracting these, we get the new Row 2: \[ \text{New Row 2} = \[0 \quad -7 \quad | \quad -18] ]

So, the matrix now looks like: \[ \begin{bmatrix} 2 & 3 & | & 8 \ 0 & -7 & | & -18 \end{bmatrix} ]

#### Step 4: Make the Leading Coefficient of the Second Row 1

We need to make the first non-zero element of the second row (which is now -7) equal to 1. To do this, multiply the entire second row by ( -\frac{1}{7} ).

\[ \text{New Row 2} = -\frac{1}{7} \times \[0 \quad -7 \quad | \quad -18] = \[0 \quad 1 \quad | \quad \frac{18}{7}] ]

The matrix is now: \[ \begin{bmatrix} 2 & 3 & | & 8 \ 0 & 1 & | & \frac{18}{7} \end{bmatrix} ]

#### Step 5: Zero Out Elements Above the Leading Coefficient of the Second Row

We want to make the second element of the first row zero. To achieve this, we can replace the first row with the result of ( \text{Row 1} - 3 \times \text{Row 2} ).

Original Row 1: ( \[2 \quad 3 \quad | \quad 8] )

( 3 \times ) New Row 2: ( \[0 \quad 3 \quad | \quad \frac{54}{7}] )

Subtracting these, we get the new Row 1: \[ \text{New Row 1} = \[2 \quad 0 \quad | \quad \frac{2}{7}] ]

#### Step 6: Make the Leading Coefficient of the First Row 1

Finally, divide the first row by 2 to make its leading coefficient 1.

\[ \text{New Row 1} = \frac{1}{2} \times \[2 \quad 0 \quad | \quad \frac{2}{7}] = \[1 \quad 0 \quad | \quad \frac{1}{7}] ]

#### Final Row Echelon Form (REF)

The final REF of the matrix is: \[ \begin{bmatrix} 1 & 0 & | & \frac{1}{7} \ 0 & 1 & | & \frac{18}{7} \end{bmatrix} ]

This REF shows that ( x = \frac{1}{7} ) and ( y = \frac{18}{7} ). This is the solution to the original system of equations.

Each step in the process involves elementary row operations, which are basic operations that can be performed on the rows of a matrix without changing its solution set. The goal is to systematically simplify the matrix until the solution can be easily read off from it.

#### What Does the Rank of a Matrix Mean?

1. **Definition**: The rank of a matrix is defined as the maximum number of linearly independent row or column vectors in the matrix. It reflects the dimension of the vector space spanned by its rows or columns.
2. **Implications**:
   * **For Systems of Equations**: In the context of a system of linear equations, the rank tells us about the solvability of the system. If the rank of the coefficient matrix equals the rank of the augmented matrix, the system is consistent; otherwise, it's inconsistent.
   * **For Dependency**: A lower rank than the maximum possible (which is the smaller of the number of rows or columns) indicates linear dependence among the rows or columns.
   * **For Inverse**: A square matrix is invertible if and only if its rank is equal to its dimension
