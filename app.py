import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp


def rref(A):
    """Compute the Reduced Row Echelon Form of matrix A."""
    A_sym = sp.Matrix(A)
    rref_matrix, _ = A_sym.rref()
    return np.array(rref_matrix).astype(float)


def check_consistency(rref_matrix):
    """Check if the system represented by the RREF matrix is consistent."""
    num_rows, num_vars = rref_matrix.shape
    return not (np.all(rref_matrix[-1, :-1] == 0) and rref_matrix[-1, -1] != 0)


def express_variables(rref_matrix, zero_vector=False):
    """Express each basic variable in terms of free variables and constants."""
    num_rows, num_vars = rref_matrix.shape

    if zero_vector:
        rref_matrix = np.hstack((rref_matrix[:, :-1], np.zeros((num_rows, 1))))

    if check_consistency(rref_matrix):
        pivot_columns = []
        for row in range(num_rows):
            pivot_col = next((i for i in range(num_vars - 1) if rref_matrix[row, i] == 1), None)
            if pivot_col is not None:
                pivot_columns.append(pivot_col)

        free_variables = [i for i in range(num_vars - 1) if i not in pivot_columns]

        expressions = {i: "" for i in range(num_vars - 1)}

        for row in range(num_rows):
            pivot_col = next((i for i in range(num_vars - 1) if rref_matrix[row, i] == 1), None)

            if pivot_col is not None:
                expression = f"x{pivot_col + 1} = {rref_matrix[row, -1]}"
                for col in range(num_vars - 1):
                    if col != pivot_col and rref_matrix[row, col] != 0:
                        sign = '-' if rref_matrix[row, col] > 0 else '+'
                        expression += f" {sign} {abs(rref_matrix[row, col])}*x{col + 1}"
                expressions[pivot_col] = expression

        result = []
        for i in range(num_vars - 1):
            if expressions[i]:
                result.append(expressions[i])

        for free_var in free_variables:
            result.append(f"x{free_var + 1} is free")

        return result
    else:
        return ["The system is inconsistent"]


def plot_solution(rref_matrix):
    """Plot solutions for both homogeneous (Ax = 0) and non-homogeneous (Ax = b) systems."""
    num_rows, num_cols = rref_matrix.shape
    num_vars = num_cols - 1  # Last column is for constants

    if num_vars < 2 or num_vars > 3:
        return "Cannot visualize systems with less than 2 or more than 3 variables."

    fig = plt.figure(figsize=(12, 6))

    if num_vars == 2:
        ax = fig.add_subplot(111)
        plot_2d(ax, rref_matrix)
    else:
        ax = fig.add_subplot(111, projection='3d')
        plot_3d(ax, rref_matrix)

    from matplotlib.lines import Line2D
    custom_lines = [Line2D([0], [0], color='blue', lw=2),
                    Line2D([0], [0], color='red', lw=2),
                    Line2D([0], [0], color='green', marker='o', linestyle='None')]
    ax.legend(custom_lines, ['Ax = 0', 'Ax = b', 'Particular Solution'])

    return fig


def plot_2d(ax, rref_matrix):
    x = np.linspace(-10, 10, 100)

    rank = np.linalg.matrix_rank(rref_matrix[:, :-1])
    num_vars = rref_matrix.shape[1] - 1
    num_free_vars = num_vars - rank

    if num_free_vars > 0:
        # Plot homogeneous solution (Ax = 0)
        a, b = rref_matrix[0, :2]
        if b != 0:
            y = (-a * x) / b
            ax.plot(x, y, color='blue', alpha=0.7)
        elif a != 0:
            ax.axvline(x=0, color='blue', alpha=0.7)

        # Plot non-homogeneous solution (Ax = b)
        c = rref_matrix[0, -1]
        if b != 0:
            y = (c - a * x) / b
            ax.plot(x, y, color='red', alpha=0.7)
        elif a != 0:
            ax.axvline(x=c / a, color='red', alpha=0.7)

    # Plot particular solution if it exists
    if num_free_vars == 0:
        particular_solution = np.linalg.lstsq(rref_matrix[:, :-1], rref_matrix[:, -1], rcond=None)[0]
        ax.plot(particular_solution[0], particular_solution[1], 'go', markersize=10)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title('Solution Sets: Ax = 0 (blue) and Ax = b (red)')
    ax.grid(True)


def plot_3d(ax, rref_matrix):
    x = y = np.linspace(-10, 10, 100)
    X, Y = np.meshgrid(x, y)

    num_vars = rref_matrix.shape[1] - 1
    rank = np.linalg.matrix_rank(rref_matrix[:, :-1])
    num_free_vars = num_vars - rank

    if num_free_vars == 1:
        # Extract coefficients and constant term
        coefs = rref_matrix[0, :-1]
        d = rref_matrix[0, -1]

        # Plot a single plane for both homogeneous and non-homogeneous solutions
        if coefs[2] != 0:
            Z = (-coefs[0] * X - coefs[1] * Y) / coefs[2]
            ax.plot_surface(X, Y, Z, alpha=0.7, color='blue')  # Adjust alpha for better visibility
        elif coefs[1] != 0:
            Y_plane = (-coefs[0] * X) / coefs[1]
            ax.plot_surface(X, Y_plane, y[:, np.newaxis], alpha=0.7, color='blue')
        elif coefs[0] != 0:
            ax.plot_surface(np.zeros_like(X), Y, X, alpha=0.7, color='blue')

        # Adjust color for non-homogeneous solution (optional)
        Z = (d - coefs[0] * X - coefs[1] * Y) / coefs[2]
        ax.plot_surface(X, Y, Z, alpha=0.5, color='red')
    # Plot particular solution if it exists
    if num_free_vars == 0:
        particular_solution = np.linalg.lstsq(rref_matrix[:, :-1], rref_matrix[:, -1], rcond=None)[0]
        ax.plot([particular_solution[0]], [particular_solution[1]], [particular_solution[2]], 'go', markersize=10)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')
    ax.set_title('Solution Sets: Ax = 0 (blue) and Ax = b (red)')


def display_matrix(matrix):
    """Display the matrix in a readable format with improved spacing, centered alignment, and increased size."""
    st.write("Matrix Representation:")
    table_html = '''
    <div style="display: flex; justify-content: center;">
        <table border="1" style="border-collapse: collapse; text-align: center; font-size: 20px; padding: 10px;">
    '''

    for row in matrix:
        table_html += '<tr>'
        for elem in row:
            table_html += f'<td style="padding: 15px; text-align: center;">{elem:.2f}</td>'
        table_html += '</tr>'

    table_html += '</table></div>'

    st.markdown(table_html, unsafe_allow_html=True)


# Main Streamlit app
st.set_page_config(page_title="Linear Algebra Solver", page_icon="🔢", layout="centered")

if 'page' not in st.session_state:
    st.session_state.page = "home"

if st.session_state.page == "home":
    st.title("Linear Algebra System Solver")
    st.write("Welcome to the Linear Algebra System Solver app.")
    st.write()
    st.write("""
        This application allows you to solve linear algebra systems by reducing them to 
        their Row Reduced Echelon Form (RREF) and expressing the solutions. You can visualize 
        the solutions for both homogeneous and non-homogeneous systems.
        """)
    st.write()
    st.image("LinearAlgebraThumbnail.png", use_column_width=True)
    st.write("Click the button below to start solving!")
    if st.button("Next"):
        st.session_state.page = "input"

elif st.session_state.page == "input":
    st.title("Matrix Dimensions Input")
    st.write("Select the number of rows and columns for the matrix.")

    cols = st.columns(2)
    with cols[0]:
        rows = st.number_input("Number of rows:", min_value=1, max_value=10, value=2, step=1)
    with cols[1]:
        cols = st.number_input("Number of columns:", min_value=1, max_value=10, value=2, step=1)

    st.write(f"Selected matrix size: {int(rows)} x {int(cols)}")

    st.write("Matrix representation:")
    matrix_repr = ""
    for i in range(rows):
        row_repr = " ".join([f"[{i + 1},{j + 1}]" for j in range(cols)])
        matrix_repr += row_repr + "\n"
    st.text(matrix_repr)

    if st.button("Done"):
        st.session_state.rows = rows
        st.session_state.cols = cols
        st.session_state.page = "elements"

elif st.session_state.page == "elements":
    st.title("Matrix Elements Input")
    st.write(f"Enter the elements for a {int(st.session_state.rows)} x {int(st.session_state.cols)} matrix.")

    matrix = np.zeros((int(st.session_state.rows), int(st.session_state.cols)))
    updated_matrix = matrix.copy()

    cols = st.columns(int(st.session_state.cols))
    for i in range(int(st.session_state.rows)):
        with st.container():
            row = []
            for j in range(int(st.session_state.cols)):
                element = cols[j].number_input(f"Element ({i + 1}, {j + 1}):", key=f"element_{i}_{j}",
                                               value=matrix[i, j])
                row.append(element)
            updated_matrix[i] = row

    display_matrix(updated_matrix)

    st.session_state.matrix = updated_matrix

    if st.button("Solve"):
        st.session_state.page = "solution"

elif st.session_state.page == "solution":
    st.title("Solution and Visualization")

    A_b = st.session_state.matrix
    A = A_b[:, :-1]
    b = A_b[:, -1]
    rref_matrix_A_b = rref(A_b)
    rref_matrix_A = rref(A)
    st.subheader("RREF of [A | b]:")
    st.write(rref_matrix_A_b)
    st.subheader("Non-homogeneous solution [A | b]:")
    non_homogeneous_sol = express_variables(rref_matrix_A_b)
    st.write(non_homogeneous_sol)
    st.subheader("RREF of A:")
    st.write(rref_matrix_A)
    st.subheader("Homogeneous solution [A | 0]:")
    homogeneous_sol = express_variables(rref_matrix_A_b, True)
    st.write(homogeneous_sol)
    fig = plot_solution(rref_matrix_A_b)
    if isinstance(fig, plt.Figure):
        st.pyplot(fig)

    if st.button("Back to Input"):
        st.session_state.page = "input"
