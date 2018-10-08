module BiharmonicSPatch

import Base: getindex, setindex!, get!
using LinearAlgebra

const Index = Vector{Int}
const Point = Vector{Float64}

struct SPatch
    n :: Int
    d :: Int
    cpts :: Dict{Index,Point}
end

getindex(s::SPatch, si::Index) = s.cpts[si]
setindex!(s::SPatch, v::Point, si::Index) = s.cpts[si] = v
get!(s::SPatch, si::Index, v::Point) = get!(s.cpts, si, v)


# Control point optimization

"""
    indices(n, d)

All possible indices of an S-patch with `n` sides and degree `d`.
"""
indices(n, d) = n == 1 ? [d] : [[i; si] for i in 0:d for si in indices(n - 1, d - i)]

"""
    pair_atleast(si, k)

Returns if there are two consecutive elements in `si` whose sum is at least `k`.
"""
function pair_atleast(si, k)
    n = length(si)
    any(i -> si[i] + si[mod1(i + 1, n)] >= k, 1:n)
end

"""
    isboundary(si)

Returns if the control point with index `si` is part of the C0 boundary.
"""
isboundary(si) = pair_atleast(si, sum(si))

"""
    isribbon(si)

Returns if the control point with index `si` is part of the G1 boundary.
"""
isribbon(si) = pair_atleast(si, sum(si) - 1)

"""
    neighbors(si)

An array of all neighbors of `si`.
"""
function neighbors(si)
    n = length(si)
    function step(i, dir)
        j = mod1(i + dir, n)
        [k == i ? x - 1 : (k == j ? x + 1 : x) for (k, x) in enumerate(si)]
    end
    valid(sj) = all(x -> x >= 0, sj)
    filter(valid, [[step(i, 1) for i in 1:n]; [step(i, -1) for i in 1:n]])
end

"""
    harmonic_mask(si)

Returns the harmonic mask as a dictionary.
"""
function harmonic_mask(si)
    neigs = neighbors(si)
    result = Dict(si => -length(neigs))
    foreach(sj -> result[sj] = 1, neigs)
    result
end

"""
    biharmonic_mask(si)

Returns the biharmonic mask as a dictionary.
"""
function biharmonic_mask(si)
    result = Dict{Index,Int}()
    for (sj, wj) in harmonic_mask(si)
        for (sk, wk) in harmonic_mask(sj)
            result[sk] = get!(result, sk, 0) + wk * wj
        end
    end
    result
end

"""
    optimize_controlnet!(surf; g1)

Modifies `surf` by placing its control points to default positions
computed by biharmonic masks.

If `g1` is true (the default), G1 boundaries are fixed.
If `g1` is false, only the C0 boundaries are fixed.
"""
function optimize_controlnet!(surf; g1 = true)
    isfixed = g1 ? isribbon : isboundary
    movable = filter(i -> !isfixed(i), indices(surf.n, surf.d))
    mapping = Dict(map(p -> (p[2], p[1]), enumerate(movable)))
    m = length(movable)
    A = zeros(m, m)
    b = zeros(m, 3)
    for (i, si) in enumerate(movable)
        for (sj, wj) in biharmonic_mask(si)
            if isfixed(sj)
                b[i,1:end] -= surf[sj] * wj
            else
                A[i,mapping[sj]] = wj
            end
        end
    end
    x = A \ b
    for (i, si) in enumerate(movable)
        surf[si] = x[i,1:end]
    end
end


# Evaluation

multinomial(xs) = factorial(sum(xs)) ÷ prod(map(factorial, xs))

bernstein(xs, bc) = multinomial(xs) * prod(map(^, bc, xs))

regularpoly(n) = [[cos(a), sin(a)] for a in range(0.0, length=n+1, stop=2pi)][1:n]

function barycentric(poly, p; barycentric_type = :wachspress, tolerance = 1.0e-5)
    vectors = map(x -> p - x, poly)
    n = length(poly)
    inc = i -> mod(i, n) + 1
    dec = i -> mod(i - 2, n) + 1
    area = i -> det([vectors[i] vectors[inc(i)]]) / 2
    area_product = exceptions -> mapreduce(area, *, setdiff(1:n, exceptions))
    f = Dict(:wachspress => x -> 1, :meanvalue => x -> norm(x), :harmonic => x -> norm(x)^2)
    a = map(i -> area_product([i]), 1:n)
    a2 = map(i -> area_product([i, dec(i)]), 1:n)
    b = map(i -> det([vectors[dec(i)] vectors[inc(i)]]) / 2, 1:n)
    r = map(f[barycentric_type], vectors)
    corner = findfirst(x -> x < tolerance, r)
    if corner != nothing
        result = fill(0, n)
        result[corner] = 1
        return result
    end
    w = [r[dec(i)] * a[dec(i)] + r[inc(i)] * a[i] - r[i] * b[i] * a2[i] for i = 1:n]
    wsum = sum(w)
    map(wi -> wi / wsum, w)
end

function eval_one(poly, surf, p)
    result = [0, 0, 0]
    bc = barycentric(poly, p)
    for (i, q) in surf.cpts
        result += q * bernstein(i, bc)
    end
    result
end

affine_combine(p, x, q) = p * (1 - x) + q * x

function vertices(poly, resolution)
    n = length(poly)
    lines = [(poly[mod1(i-1,n)], poly[i]) for i in 1:n]
    center = [0.0, 0.0]
    result = [center]
    for j in 1:resolution
        coeff = j / resolution
        for k in 1:n, i in 0:j-1
            lp = affine_combine(lines[k][1], i / j, lines[k][2])
            push!(result, affine_combine(center, coeff, lp))
        end
    end
    result
end

function triangles(n, resolution)
    result = []
    inner_start = 1
    outer_vert = 2
    for layer in 1:resolution
        inner_vert = inner_start
        outer_start = outer_vert
        for side in 1:n
            vert = 1
            while true
                next_vert = side == n && vert == layer ? outer_start : outer_vert + 1
                push!(result, [inner_vert, outer_vert, next_vert])
                outer_vert += 1
                vert += 1
                vert == layer + 1 && break
                inner_next = side == n && vert == layer ? inner_start : inner_vert + 1
                push!(result, [inner_vert, next_vert, inner_next])
                inner_vert = inner_next
            end
        end
        inner_start = outer_start
    end
    result
end

function eval_all(surf, resolution)
    poly = regularpoly(surf.n)
    ([eval_one(poly, surf, p) for p in vertices(poly, resolution)],
     triangles(surf.n, resolution))
end


# C0 ribbon computation

const GBIndex = NTuple{3, Int}

struct BezierPatch
    n :: Int
    d :: Int
    cpts :: Dict{GBIndex,Point}
end

getindex(s::BezierPatch, idx::GBIndex) = s.cpts[idx]
getindex(s::BezierPatch, i::Int, j::Int, k::Int) = s.cpts[i,j,k]
setindex!(s::BezierPatch, v::Point, idx::GBIndex) = s.cpts[idx] = v
setindex!(s::BezierPatch, v::Point, i::Int, j::Int, k::Int) = s.cpts[i,j,k] = v

function make_index(n, pairs...)
    index = zeros(Int, n)
    for (i, v) in pairs
        index[mod1(i,n)] = v
    end
    index
end

function c0_patch(ribbons)
    n, d = ribbons.n, ribbons.d
    result = SPatch(n, d, Dict())
    for i in 0:n-1, j in 0:d
        index = make_index(n, (i+1,d-j), (i+2,j))
        result[index] = ribbons[i,j,0]
    end
    result
end


# G1 ribbon computation

function affine_image(points)
    poly = regularpoly(length(points))
    A = [poly[1]' 1; poly[2]' 1; poly[3]' 1]
    b = [points[1]'; points[2]'; points[3]']
    M = (A \ b)'
    map(p -> M * [p; 1], poly)
end

function panel_indices(n, d, i, j)
    index = make_index(n, (i,d-j), (i+1,j))
    map(1:n) do j
        result = copy(index)
        index[mod1(i+j-1, n)] -= 1
        index[mod1(i+j, n)] += 1
        result
    end
end

function set_panels!(surf)
    n, d = surf.n, surf.d
    for i in 1:n, j in 0:d-1
        idxs = panel_indices(n, d, i, j)
        points = affine_image(map(idx -> get!(surf, idx, Point()), idxs))
        foreach((idx, p) -> surf[idx] = p, idxs, points)
    end
end

function g1normals(ribbons, i)
    i -= 1
    d = ribbons.d
    result = []
    function add_normal!(j, p)
        o = ribbons[i,j,0]
        q = ribbons[i,j+1,0]
        push!(result, normalize!(cross(q - o, p - o)))
    end
    prev = ribbons[i,0,1]
    add_normal!(0, prev)
    for j in 1:d-1
        c = j / d
        mid = affine_combine(ribbons[i,j,0], c, ribbons[i,j-1,0])
        next = (ribbons[i,j,1] - ribbons[i,j,0] - c * prev + mid) / (1 - c)
        add_normal!(j, next)
        prev = next
    end
    result
end

function panel_leg(surf, normal, i, j)
    n, d = surf.n, surf.d
    left_corner = make_index(n, (i,d))
    left = make_index(n, (i-1,1), (i,d-1))
    right = make_index(n, (i,d-1), (i+1,1))
    left_panel = map(idx -> surf[idx], [left, left_corner, right])
    pleft = left_panel[3]
    pleftleg = affine_image([left_panel; Vector(undef, n - 3)])[4]
    
    right_corner = make_index(n, (i+1,d))
    right_leg = make_index(n, (i+1,d-1), (i+2,1))
    pright = surf[right_corner]
    prightleg = surf[right_leg]

    current = make_index(n, (i,d-j), (i+1,j))
    pcurrent = surf[current]

    c = (j - 1) / (d - 1)
    v = affine_combine(pleftleg - pleft, c, prightleg - pright)
    pcurrent + v - normal * dot(v, normal)
end

function set_g1_continuity!(surf, ribbons)
    n, d = surf.n, surf.d
    for i in 1:n
        normals = g1normals(ribbons, i)
        for j in 1:d
            index = make_index(n, (i,d-j), (i+1,j-1), (i+2,1))
            surf[index] = panel_leg(surf, normals[j], i, j)
        end
    end
    set_panels!(surf)
end


# Output

function write_cnet(surf, filename; g1 = true, only_fixed = false)
    isfixed = g1 ? isribbon : isboundary
    mapping = Dict(map(p -> (p[2], p[1]), enumerate(keys(surf.cpts))))
    open(filename, "w") do f
        for p in values(surf.cpts)
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        for i in keys(surf.cpts)
            only_fixed && !isfixed(i) && continue
            from = mapping[i]
            for j in neighbors(i)
                only_fixed && !isfixed(j) && continue
                to = mapping[j]
                from < to && println(f, "l $from $to")
            end
        end
    end
end

function writeOBJ(verts, tris, filename)
    open(filename, "w") do f
        for v in verts
            println(f, "v $(v[1]) $(v[2]) $(v[3])")
        end
        for t in tris
            println(f, "f $(t[1]) $(t[2]) $(t[3])")
        end
    end
end

function write_surface(surf, filename, resolution)
    vs, ts = eval_all(surf, resolution)
    writeOBJ(vs, ts, filename)
end


# Bezier ribbon I/O & evaluation

function read_ribbons(filename)
    read_numbers(f, numtype) = map(s -> parse(numtype, s), split(readline(f)))
    local result
    open(filename) do f
        n, d = read_numbers(f, Int)
        result = BezierPatch(n, d, Dict())
        l = Int(floor(d + 1) / 2)
        cp = 1 + Int(floor(d / 2))
        cp = n * cp * l + 1
        side = 0
        col = 0
        row = 0
        readline(f)
        for i in 1:cp-1
            if col >= d - row
                side += 1
                if side >= n
                    side = 0
                    row += 1
                end
                col = row
            end
            p = read_numbers(f, Float64)
            result[side,col,row] = p
            if col < l
                result[mod(side-1,n),d-row,col] = p
            elseif d - col < l
                result[mod(side+1,n),row,d-col] = p
            end
            col += 1
        end
    end
    result
end

function write_ribbon(ribbons, filename, index, resolution)
    d = ribbons.d
    bezier(n, k, x) = binomial(n, k) * x ^ k * (1 - x) ^ (n - k)
    samples = [[u, v] for u in range(0.0, stop=1.0, length=resolution)
                      for v in range(0.0, stop=1.0, length=resolution)]
    verts = map(samples) do p
        result = [0, 0, 0]
        for i in 0:d, j in 0:1
            result += ribbons[index,i,j] * bezier(d, i, p[1]) * bezier(1, j, p[2])
        end
        result
    end
    tris = []
    for i in 2:resolution, j in 2:resolution
        index = (j - 1) * resolution + i
        push!(tris, [index - resolution - 1, index - resolution, index])
        push!(tris, [index, index - 1, index - resolution - 1])
    end
    writeOBJ(verts, tris, filename)
end

function write_bezier_cnet(ribbons, filename)
    n, d = ribbons.n, ribbons.d
    g1 = filter(idx -> idx[3] <= 1, keys(ribbons.cpts))
    mapping = Dict(map(p -> (p[2], p[1]), enumerate(g1)))
    open(filename, "w") do f
        for idx in g1
            p = ribbons[idx]
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        for idx in g1
            i, j, k = idx
            from = mapping[idx]
            for next in [(i, j + 1, k), (i, j, k + 1)]
                !haskey(mapping, next) && continue
                to = mapping[next]
                println(f, "l $from $to")
            end
        end
    end
end


# Main function

function spatch_test(name, resolution, g1_continuity = true)
    ribbons = read_ribbons("$name.gbp")
    surf = c0_patch(ribbons)
    g1_continuity && set_g1_continuity!(surf, ribbons)
    optimize_controlnet!(surf, g1 = g1_continuity)
    write_surface(surf, "$name.obj", resolution)
    write_cnet(surf, "$name-cnet.obj")
    write_cnet(surf, "$name-ribbon.obj", g1 = g1_continuity, only_fixed = true)
    g1_continuity && write_bezier_cnet(ribbons, "$name-bezier-cnet.obj")
end

end # module
