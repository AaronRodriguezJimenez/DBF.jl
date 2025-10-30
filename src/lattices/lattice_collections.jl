#
# - - -  Square Lattices - - - 
#
# Helper: construct bond in canonical (i<j) order
function _pushbond!(bonds::Vector{LatticeBond}, i::Int, j::Int, xi::Int, yi::Int, xj::Int, yj::Int)
    if i == j
        return
    end
    if i < j
        push!(bonds, LatticeBond(i, j, xi, yi, xj, yj))
    else
        push!(bonds, LatticeBond(j, i, xj, yj, xi, yi))
    end
end

"""
build_square_lattice(Nx, Ny; layout=:standard, order=:row, xperiodic=false, yperiodic=false)

Returns a Vector{LatticeBond} for a Nx-by-Ny square lattice.

Arguments:
- Nx, Ny : positive integers grid dimensions
- layout : :standard or :zigzag (default :standard)
- order  : :row or :col — which axis is the fast index in the 1D flattening
           :row -> n = (y-1)*Nx + x  (row-major)
           :col -> n = (x-1)*Ny + y  (column-major)
- xperiodic, yperiodic : booleans for periodic boundary in x or y

Returns: (bonds::Vector{LatticeBond}, idx::Function)
- bonds : canonical undirected bond list
- idx(site_x, site_y) -> linear index (1..Nx*Ny) according to chosen order
"""
function build_square_lattice(Nx::Int, Ny::Int; layout::Symbol=:standard,
                              order::Symbol=:row, xperiodic::Bool=false,
                              yperiodic::Bool=false)

    @assert Nx >= 1 && Ny >= 1 "Nx and Ny must be >= 1"
    @assert layout in (:standard, :zigzag) "layout must be :standard or :zigzag"
    @assert order in (:row, :col) "order must be :row or :col"

    # index mapping functions for row- or column-major flattening
    if order == :row
        idx = (x,y) -> (y-1)*Nx + x
        idx_zigzag = (x,y) -> begin
            if isodd(y)
                return (y-1)*Nx + x
            else
                return (y-1)*Nx + (Nx - x + 1)
            end
        end
    else # :col
        idx = (x,y) -> (x-1)*Ny + y
        idx_zigzag = (x,y) -> begin
            if isodd(x)
                return (x-1)*Ny + y
            else
                return (x-1)*Ny + (Ny - y + 1)
            end
        end
    end

    bonds = Vector{LatticeBond}()

    # Determine the bond order for the lattice:
    # If layout==:standard, bond order runs in a standard grid fashion.
    # If layout==:zigzag, bond order is created in a zigzag pattern.
    
    # choose which axis is fast (inner loop) and which is slow (outer)
    if order == :row
        # fast = x, slow = y
        fast_range = 1:Nx
        slow_range = 1:Ny
        fast_axis = :x
    else
        # fast = y, slow = x
        fast_range = 1:Ny
        slow_range = 1:Nx
        fast_axis = :y
    end

    for slow in slow_range
        # compute direction for this slow-line (for zigzag)
        forward = true
        if layout == :zigzag
            # alternate direction: even slow index -> reverse (or choose pattern)
            # Use parity of slow to alternate: odd -> forward, even -> backward
            forward = isodd(slow)
        end

        fiter = forward ? fast_range : reverse(fast_range)
        for fast in fiter
            # map back to (x,y)
            if order == :row
                x, y = fast, slow
            else
                x, y = slow, fast
            end

            if layout == :zigzag
                n = idx_zigzag(x, y)
            else
                n = idx(x, y)
            end
            

            # neighbor in fast direction (within same slow-line)
            if forward
                # attempt link to next fast (forward)
                if fast < last(fast_range)
                    f2 = fast + 1
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                elseif (layout == :zigzag) && (forward) && false
                    # no-op: we don't automatically wrap along fast axis unless periodicities enabled
                end
            else
                # reversed direction, attempt link to previous fast
                if fast > first(fast_range)
                    f2 = fast - 1
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                end
            end

            # neighbor in slow direction (connect to next slow line, same fast position)
            # compute target slow'
            if slow < last(slow_range)
                slow2 = slow + 1
                if order == :row
                    x2, y2 = x, slow2
                else
                    x2, y2 = slow2, y
                end
                if layout == :zigzag
                    n2 = idx_zigzag(x2, y2)
                else
                    n2 = idx(x2, y2)
                end

                _pushbond!(bonds, n, n2, x, y, x2, y2)
            elseif slow == last(slow_range) && ( (order==:row && yperiodic) || (order==:col && xperiodic) )
                # wrap periodic in slow direction:
                slow2 = first(slow_range)
                if order == :row
                    x2, y2 = x, slow2
                else
                    x2, y2 = slow2, y
                end
                if layout == :zigzag
                    n2 = idx_zigzag(x2, y2)
                else
                    n2 = idx(x2, y2)
                end
                _pushbond!(bonds, n, n2, x, y, x2, y2)
            end

            # horizontal / fast-direction periodic wrap if enabled and at boundary
            # (this handles periodicity across fast axis boundaries)
            if forward && (fast == last(fast_range))
                # at end of fast axis in forward direction
                if (order == :row && xperiodic) || (order == :col && yperiodic)
                    # wrap to first fast in same slow
                    f2 = first(fast_range)
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                end
            elseif (!forward) && (fast == first(fast_range))
                # at start in reversed direction and periodic desired
                if (order == :row && xperiodic) || (order == :col && yperiodic)
                    f2 = last(fast_range)
                    if order == :row
                        x2, y2 = f2, slow
                    else
                        x2, y2 = slow, f2
                    end
                    if layout == :zigzag
                        n2 = idx_zigzag(x2, y2)
                    else
                        n2 = idx(x2, y2)
                    end
                    _pushbond!(bonds, n, n2, x, y, x2, y2)
                end
            end
        end
    end

    # sanity check: expected undirected bond count (no double-count)
    expected = (Nx-1)*Ny + Nx*(Ny-1) + (xperiodic ? Ny : 0) + (yperiodic ? Nx : 0)

    return bonds#, idx
end

"""
    Standard square_lattice(Nx::Int,
                   Ny::Int;
                   kwargs...)::Lattice

Return a Lattice (array of LatticeBond
objects) corresponding to the two-dimensional
square lattice of dimensions (Nx,Ny).
By default the lattice has open boundaries,
but can be made periodic in the y direction
by specifying the keyword argument
`yperiodic=true`.
"""
function square_lattice(Nx::Int, Ny::Int; yperiodic=false)::Lattice
  yperiodic = yperiodic && (Ny > 2)
  N = Nx * Ny
  Nbond = 2N - Ny + (yperiodic ? 0 : -Nx)
  latt = Lattice(undef, Nbond)
  b = 0
  for n in 1:N
    x = div(n - 1, Ny) + 1
    y = mod(n - 1, Ny) + 1
    if x < Nx
      latt[b += 1] = LatticeBond(n, n + Ny, x, y, x + 1, y)
    end
    if Ny > 1
      if y < Ny
        latt[b += 1] = LatticeBond(n, n + 1, x, y, x, y + 1)
      end
      if yperiodic && y == 1
        latt[b += 1] = LatticeBond(n, n + Ny - 1, x, y, x, y + Ny - 1)
      end
    end
  end
  return latt
end

"""
    triangular_lattice(Nx::Int,
                       Ny::Int;
                       kwargs...)::Lattice

Return a Lattice (array of LatticeBond
objects) corresponding to the two-dimensional
triangular lattice of dimensions (Nx,Ny).
By default the lattice has open boundaries,
but can be made periodic in the y direction
by specifying the keyword argument
`yperiodic=true`.
"""
function triangular_lattice(Nx::Int, Ny::Int; yperiodic=false)::Lattice
  yperiodic = yperiodic && (Ny > 2)
  N = Nx * Ny
  Nbond = 3N - 2Ny + (yperiodic ? 0 : -2Nx + 1)
  latt = Lattice(undef, Nbond)
  b = 0
  for n in 1:N
    x = div(n - 1, Ny) + 1
    y = mod(n - 1, Ny) + 1

    # x-direction bonds
    if x < Nx
      latt[b += 1] = LatticeBond(n, n + Ny)
    end

    # 2d bonds
    if Ny > 1
      # vertical / y-periodic diagonal bond
      if (n + 1 <= N) && ((y < Ny) || yperiodic)
        latt[b += 1] = LatticeBond(n, n + 1)
      end
      # periodic vertical bond
      if yperiodic && y == 1
        latt[b += 1] = LatticeBond(n, n + Ny - 1)
      end
      # diagonal bonds
      if x < Nx && y < Ny
        latt[b += 1] = LatticeBond(n, n + Ny + 1)
      end
    end
  end
  return latt
end