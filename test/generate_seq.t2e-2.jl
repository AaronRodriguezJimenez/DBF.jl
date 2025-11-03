using DBF
using PauliOperators
using Printf
using Random
using LinearAlgebra
using JLD2

function run()
    N = 100 
    H = DBF.heisenberg_1D(N, -1, -1, -1)
    DBF.coeff_clip!(H)
    
    g,a = DBF.get_1d_neel_state_sequence(N)
    g,a = DBF.get_rvb_sequence(N)
   
    for (gi, ai) in zip(g,a)
        H = evolve(H, gi, ai)
    end
   
    ψ = Ket{N}(0)
    display(ψ)
    e0 = expectation_value(H,ψ)
  
    @printf(" Reference = %12.8f\n", e0)
    flush(stdout) 
    println("\n ########################")
    @time res = DBF.dbf_groundstate(H, ψ,
                                    verbose=1,
                                    max_iter=100, 
                                    conv_thresh=1e-3,
                                    evolve_coeff_thresh=1e-2,
                                    grad_coeff_thresh=1e-6,
                                    energy_lowering_thresh=1e-6,
                                    max_rots_per_grad=100, 
                                    checkfile = "t1e-2")
    
    
end


run()