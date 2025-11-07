# # Test Hubbard 1D
# H = hubbard_model_1D(2, 5.0, 2.0)
# display(H)
# println("Number of terms in Hubbard 1D Hamiltonian: ", length(H))
using DBF

# # Test Hubbard 2D
H = DBF.fermi_hubbard_2D(3, 3, 5.0, 2.0)
display(H)
println("Number of terms in Hubbard 2x2 Hamiltonian: ", length(H))

# # Test Hubbard 2D zigzag
H_zigzag = DBF.fermi_hubbard_2D_zigzag(3, 3, 5.0, 2.0)
display(H_zigzag)
println("Number of terms in Hubbard 2x2 zigzag Hamiltonian: ", length(H_zigzag)) 

# # Test Hubbard 2D snake
H_snake = DBF.fermi_hubbard_2D_snake(3, 3, 5.0, 2.0, snake_ordering=true)
display(H_snake)
println("Number of terms in Hubbard 2x2 snake Hamiltonian: ", length(H_snake))
#

# # Test Hubbard from lattice
H_lattice = DBF.fermi_hubbard_from_lattice(3, 3, 5.0, 2.0)
display(H_lattice)
println("Number of terms in Hubbard from lattice Hamiltonian: ", length(H_lattice))

