# Week 2: Solving system of linear equations

1\. Graphical Method

* **Description**: Plot each equation as a line on a graph. The point(s) where the lines intersect are the solutions.
* **Example**:
  * Equations: ( y = 2x + 1 ) and ( y = -x + 3 )
  * Plot these lines on a graph to find the intersection point.

#### 2. Substitution Method

* **Description**: Solve one equation for one variable and substitute this value in the other equation.
* **Example**:
  * Equations: ( x + y = 5 ) and ( x - y = 1 )
  * Solve the first equation for ( x ): ( x = 5 - y )
  * Substitute in the second equation: ( 5 - y - y = 1 )

#### 3. Elimination Method

* **Description**: Add or subtract equations to eliminate one variable, then solve for the other.
* **Example**:
  * Equations: ( 2x + 3y = 7 ) and ( 4x - y = 1 )
  * Multiply the second equation by 3 and add to the first to eliminate ( y ).

#### 4. Matrix Method (Gaussian Elimination)

* **Description**: Convert equations to matrix form and use row operations to find solutions.
* **Example**:
  * Equations: ( 2x + 3y = 5 ) and ( 4x + 6y = 10 )
  * Matrix form: \[ \begin{pmatrix} 2 & 3 \ 4 & 6 \end{pmatrix} \begin{pmatrix} x \ y \end{pmatrix} = \begin{pmatrix} 5 \ 10 \end{pmatrix} ]
  * Apply row operations to solve.

#### 5. Cramer's Rule

* **Description**: Use determinants to solve each variable. Applicable when there are as many equations as variables.
* **Example**:
  * Equations: ( ax + by = e ) and ( cx + dy = f )
  * Solution for ( x ): ( x = \frac{\begin{vmatrix} e & b \ f & d \end{vmatrix\}}{\begin{vmatrix} a & b \ c & d \end{vmatrix\}} )

#### 6. Inverse Matrix Method

* **Description**: Use the inverse of the coefficient matrix to solve the system.
* **Example**:
  * Matrix form: ( AX = B )
  * Solution: ( X = A^{-1}B )

#### 7. Iterative Methods (Jacobi, Gauss-Seidel)

* **Description**: Start with an initial guess and iteratively refine the solution.
* **Example**: Not easily represented in a simple Markdown/LaTeX format, as it involves algorithmic steps.

#### 8. Numerical Methods (Newton's Method)

* **Description**: Use approximations to find solutions when exact methods are difficult.
* **Example**: Typically implemented using programming and not easily represented in Markdown/LaTeX.

These methods provide a range of tools for solving linear systems, from simple two-variable problems to complex systems with many variables. The choice of method depends on the specifics of the problem, the size of the system, and the available computational resources.
