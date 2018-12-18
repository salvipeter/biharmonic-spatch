module BiharmonicSPatch

import Base: getindex, setindex!, get!
using LinearAlgebra

const FactInt = Int64 # BigInt
const Index = Vector{Int}
const Point = Vector{Float64}

"""
    SPatch(n, d, cpts)

Represents an S-patch of degree `d` with `n` sides.
Its control points are stored as a dictionary,
and can be directly accessed by indexing.
"""
struct SPatch
    n :: Int
    d :: Int
    cpts :: Dict{Index,Point}
end

getindex(s::SPatch, si::Index) = s.cpts[si]
setindex!(s::SPatch, p::Point, si::Index) = s.cpts[si] = p
get!(s::SPatch, si::Index, p::Point) = get!(s.cpts, si, p)


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
    mask = Dict(si => -length(neigs))
    foreach(sj -> mask[sj] = 1, neigs)
    mask
end

"""
    biharmonic_mask(si)

Returns the biharmonic mask as a dictionary.
"""
function biharmonic_mask(si)
    mask = Dict{Index,Int}()
    for (sj, wj) in harmonic_mask(si)
        for (sk, wk) in harmonic_mask(sj)
            mask[sk] = get!(mask, sk, 0) + wk * wj
        end
    end
    mask
end

"""
    optimize_controlnet!(surf; g1)

Modifies `surf` by placing its control points to default positions
computed by biharmonic masks.

If `g1` is `true` (the default), G1 boundaries are fixed.
If `g1` is `false`, only the C0 boundaries are fixed.
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

"""
    fact(n)

Arbitrary-length factorial.
"""
fact(n) = prod(1:FactInt(n))

"""
    multinomial(si)

The multinomial value of `sum(si)` over the elements of `si`.
"""
multinomial(si) = fact(sum(si)) ÷ prod(map(fact, si))

"""
    bernstein(si, bc)

The Bernstein polynomial associated with `si` evaluated at
the barycentric coordinates `bc`.
"""
bernstein(si, bc) = multinomial(si) * prod(map(^, bc, si))

"""
    regularpoly(n)

A regular `n`-gon on the inscribed circle of [0,1]x[0,1], consisting of an array of points.
For `n = 4` it returns the whole [0,1]x[0,1] square.
"""
function regularpoly(n)
    if n == 4
        [[0., 0], [1, 0], [1, 1], [0, 1]]
    else
        [[0.5+cos(a)/2, 0.5+sin(a)/2] for a in range(0.0, length=n+1, stop=2pi)][1:n]
    end
end

"""
    barycentric(poly, p; barycentric_type, tolerance)

The barycentric coordinates of `p` inside the polygon `poly`.

The coordinate type `barycentric_type` can be
- :wachspress (Wachspress coordinates)
- :meanvalue (Mean value coordinates)
- :harmonic (Discrete harmonic coordinates)

The parameter `tolerance` controls the minimum valid distance from a corner.
"""
function barycentric(poly, p; barycentric_type = :wachspress, tolerance = 1.0e-5)
    vectors = map(x -> p - x, poly)
    n = length(poly)
    inc = i -> mod1(i + 1, n)
    dec = i -> mod1(i - 1, n)
    area = i -> det([vectors[i] vectors[inc(i)]]) / 2
    area_product = exceptions -> mapreduce(area, *, setdiff(1:n, exceptions))
    f = Dict(:wachspress => x -> 1, :meanvalue => x -> norm(x), :harmonic => x -> norm(x)^2)
    a = map(i -> area_product([i]), 1:n)
    a2 = map(i -> area_product([i, dec(i)]), 1:n)
    b = map(i -> det([vectors[dec(i)] vectors[inc(i)]]) / 2, 1:n)
    r = map(f[barycentric_type], vectors)
    corner = findfirst(x -> x < tolerance, r)
    if corner != nothing
        bc = zeros(n)
        bc[corner] = 1
        return bc
    end
    w = [r[dec(i)] * a[dec(i)] + r[inc(i)] * a[i] - r[i] * b[i] * a2[i] for i = 1:n]
    wsum = sum(w)
    map(wi -> wi / wsum, w)
end

"""
    eval_one(surf, poly, p)

Evaluates the S-patch `surf` at domain point `p` inside the domain polygon `poly`.
"""
function eval_one(surf, poly, p)
    result = [0, 0, 0]
    bc = barycentric(poly, p)
    for (i, q) in surf.cpts
        result += q * bernstein(i, bc)
    end
    result
end

"""
    affine_combine(p, x, q)

Computes the linear combination between `p` and `q` at ratio `x`.
"""
affine_combine(p, x, q) = p * (1 - x) + q * x

"""
    vertices(poly, resolution)

An array of vertices sampled from the inside and boundary of `poly`.
The points are taken in a way similar to a spider's web,
with `resolution + 1` points at the boundaries.
"""
function vertices(poly, resolution)
    n = length(poly)
    lines = [(poly[mod1(i-1,n)], poly[i]) for i in 1:n]
    center = [0.5, 0.5]
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

"""
    triangles(n, resolution)

An array of index-triples connecting the points of the output of `vertices`
into a consistent triangulation.
"""
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

"""
    eval_all(surf, resolution)

Evaluates the S-patch `surf` at a given `resolution` (interpreted as in `vertices`),
and returns a pair of arrays (vertices, triangles).
"""
function eval_all(surf, resolution)
    poly = regularpoly(surf.n)
    ([eval_one(surf, poly, p) for p in vertices(poly, resolution)],
     triangles(surf.n, resolution))
end


# C0 ribbon computation

const GBIndex = NTuple{3, Int}

"""
    BezierPatch(n, d, cpts)

Represents a GB-patch of degree `d` with `n` sides.
Its control points are stored as a dictionary,
and can be directly accessed by indexing.
"""
struct BezierPatch
    n :: Int
    d :: Int
    cpts :: Dict{GBIndex,Point}
end

getindex(s::BezierPatch, idx::GBIndex) = s.cpts[idx]
getindex(s::BezierPatch, i::Int, j::Int, k::Int) = s.cpts[i,j,k]
setindex!(s::BezierPatch, v::Point, idx::GBIndex) = s.cpts[idx] = v
setindex!(s::BezierPatch, v::Point, i::Int, j::Int, k::Int) = s.cpts[i,j,k] = v

"""
    make_index(n, pairs...)

Creates an index of length `n`, with non-zero elements at specified places.

    julia> make_index(5, (2,3), (3,1))
    [0, 3, 1, 0, 0]
"""
function make_index(n, pairs...)
    si = zeros(Int, n)
    for (i, x) in pairs
        si[mod1(i,n)] = x
    end
    si
end

"""
    c0_patch(ribbons)

Creates an S-patch that has the same C0 boundaries as the given
GB-patch `ribbons`. All inner control points are undefined.
"""
function c0_patch(ribbons)
    n, d = ribbons.n, ribbons.d
    surf = SPatch(n, d, Dict())
    for i in 0:n-1, j in 0:d
        si = make_index(n, (i+1,d-j), (i+2,j))
        surf[si] = ribbons[i,j,0]
    end
    surf
end


# G1 ribbon computation

"""
    affine_image(points)

Takes an array of n 3D points, and returns another
that is the affine image of a regular n-gon,
with the first 3 points being unchanged.
"""
function affine_image(points)
    poly = regularpoly(length(points))
    A = [poly[1]' 1; poly[2]' 1; poly[3]' 1]
    b = [points[1]'; points[2]'; points[3]']
    M = (A \ b)'
    map(p -> M * [p; 1], poly)
end

"""
    panel_indices(n, d, i, j)

Gives the indices of a G1 boundary panel in an `n`-sided S-patch
of degree `d`.

This is the `j`-th panel is on the `i`-th side,
i.e., on the side where the endpoints have indices
with a non-zero element on the `i`-th and `(i+1)`-th position.

Note that `i` is between `1` and `n`, while `j` is between `0` and `d-1`.
"""
function panel_indices(n, d, i, j)
    si = make_index(n, (i,d-j-1), (i+1,j+1))
    map(1:n) do j
        result = copy(si)
        si[mod1(i+2-j, n)] -= 1
        si[mod1(i+1-j, n)] += 1
        result
    end
end

"""
    set_panels!(surf)

Modifies `surf` by setting its G1 boundary panels to be the
affine image of a regular n-sided polygon.
It is assumed that the first three points (the two on the boundary, and
the one with the index pattern ...01XY0...) of each panel is already set.
"""
function set_panels!(surf)
    n, d = surf.n, surf.d
    for i in 1:n, j in 0:d-1
        panel = panel_indices(n, d, i, j)
        points = affine_image(map(si -> get!(surf, si, Point()), panel))
        foreach((si, p) -> surf[si] = p, panel, points)
    end
end

"""
    elevated_points(points)

Assuming that `points` is the control polygon of a Bezier curve,
the function returns its degree-elevated version.
"""
function elevated_points(points)
    d = length(points) - 1
    result = [points[1]]
    for i in 1:d
        p = affine_combine(points[i], (d + 1 - i) / (d + 1), points[i+1])
        push!(result, p)
    end
    push!(result, points[end])
    result
end

"""
    panel_legs(ribbons, i)

Computes the "legs" of all panels on side `i`, i.e.,
the deviation vector of the control points of the form
[...01XY0...] - [...0(X+1)Y0...], in such a way that the
resulting patch will have the same G1 cross-derivative
properties as `ribbons` at the same side.

Note that `i` is between `0` and `n-1`.
"""
function panel_legs(ribbons, i)
    n, d = ribbons.n, ribbons.d
    result = []
    c = -cos(2pi / n)
    for k in 0:d+2
        p = [0.0, 0.0, 0.0]
        function add_coeff!(lo, hi, scale, du)
            if lo <= k <= hi
                j = k - lo
                v = (du ? ribbons[i,j+1,0] : ribbons[i,j,1]) - ribbons[i,j,0]
                p += v * scale * binomial(du ? d - 1 : d, j)
            end
        end
        add_coeff!(1, d,     2c,     true)
        add_coeff!(2, d + 1, 4c,     true)
        add_coeff!(3, d + 2, 2c,     true)
        add_coeff!(0, d,     1,      false)
        add_coeff!(1, d + 1, 2 + 2c, false)
        add_coeff!(2, d + 2, 1,      false)
        push!(result, p / binomial(d + 2, k))
    end
    result * d / (d + 3)
end

"""
    g1_patch(ribbons)

Creates an S-patch that has the same G1 boundaries as the given
GB-patch `ribbons`. All inner control points are undefined.

The resulting surface will be 3 degrees higher than the GB-patch.
"""
function g1_patch(ribbons)
    n, d = ribbons.n, ribbons.d
    surf = SPatch(n, d + 3, Dict())
    for i in 0:n-1
        points = [ribbons[i,j,0] for j in 0:d]
        for j in 1:3
            points = elevated_points(points)
        end
        for j in 0:d+3
            si = make_index(n, (i+1,d+3-j), (i+2,j))
            surf[si] = points[j+1]
        end
        legs = panel_legs(ribbons, i)
        for j in 1:d+2
            si = make_index(n, (i,1), (i+1,d+2-j), (i+2,j))
            surf[si] = points[j+1] + legs[j+1]
        end
    end
    set_panels!(surf)
    surf
end


# Output

"""
    write_cnet(surf, filename; g1, only_fixed)

Writes the control network of the S-patch `surf` into a Wavefront Object file
designated by `filename`. If `only_fixed` is `true`, it only writes
the control points associated with the boundary constraints -
the boundary curves, when `g1` is `false`, and the boundary panels,
when `g1` is `true`.
"""
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

"""
    writeOBJ(verts, tris, filename; normals)

Writes the mesh defined by vertices `verts` and triangles `tris`
into the Wavefront Object file designated by `filename`.

Normals can be added with the same indexing as the vertices.
"""
function writeOBJ(verts, tris, filename; normals = [])
    open(filename, "w") do f
        for v in verts
            println(f, "v $(v[1]) $(v[2]) $(v[3])")
        end
        if normals == []
            for t in tris
                println(f, "f $(t[1]) $(t[2]) $(t[3])")
            end
        else
            for n in normals
                println(f, "vn $(n[1]) $(n[2]) $(n[3])")
            end
            for t in tris
                println(f, "f $(t[1])//$(t[1]) $(t[2])//$(t[2]) $(t[3])//$(t[3])")
            end
        end
    end
end

"""
    write_surface(surf, filename, resolution)

Writes the S-patch `surf` into a Wavefront Object file designated by
`filename` with sampling rate `resolution` (interpreted as in `vertices`).
"""
function write_surface(surf, filename, resolution)
    vs, ts = eval_all(surf, resolution)
    writeOBJ(vs, ts, filename)
end


# Bezier ribbon I/O & evaluation

"""
    read_ribbons(filename)

Reads a GB-patch from a GBP file designated by `filename`.
"""
function read_ribbons(filename)
    read_numbers(f, numtype) = map(s -> parse(numtype, s), split(readline(f)))
    local result
    open(filename) do f
        n, d = read_numbers(f, Int)
        result = BezierPatch(n, d, Dict())
        l = Int(floor((d + 1) / 2))
        cp = 1 + Int(floor(d / 2))
        cp = n * cp * l + 1
        side, col, row = 0, 0, 0
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

"""
    write_ribbon(ribbons, filename, index, resolution)

Write the `index`-th ribbon of the GB-patch `ribbons` into the Wavefront Object
file designated by `filename`, with `resolution` x `resolution` sampled points.
"""
function write_ribbon(ribbons, filename, index, resolution)
    bezier(n, k, x) = binomial(n, k) * x ^ k * (1 - x) ^ (n - k)
    samples = [[u, v] for u in range(0.0, stop=1.0, length=resolution)
                      for v in range(0.0, stop=1.0, length=resolution)]
    d = ribbons.d
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

"""
    write_bezier_cnet(ribbons, filename)

Writes the ribbon structure of the GB-patch `ribbons` into the
Wavefront Object file designated by `filename`.
"""
function write_bezier_cnet(ribbons, filename)
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

"""
    bernstein_all(n, u)

Evaluates all Bernstein polynomials of degree `n` at `u`.
The result is a vector of `n+1` elements.
"""
function bernstein_all(n, u)
    coeff = [1.0]
    for j in 1:n
        saved = 0.0
        for k in 1:j
            tmp = coeff[k]
            coeff[k] = saved + tmp * (1.0 - u)
            saved = tmp * u
        end
        push!(coeff, saved)
    end
    coeff
end

"""
    evaltensor(surf, uv)

Evaluates a tensor product rational Bézier surface at the given parameters.
The surface is given as 3D array where `surf[i,j,:]` is one control point.
"""
function evaltensor(surf, uv)
    d = size(surf, 1) - 1
    coeff_u = bernstein_all(d, uv[1])
    coeff_v = bernstein_all(d, uv[2])
    result = [0, 0, 0, 0]
    for i in 0:d, j in 0:d
        result += surf[i+1,j+1,:] * coeff_u[i+1] * coeff_v[j+1]
    end
    result[1:3] / result[4]
end

"""
    write_tensor(surf, n, filename, resolution)

Writes a tensor product Bézier surface to a mesh file with the given resolution.
When `n = 0`, the whole domain is taken; otherwise only an `n`-sided domain is sampled.
"""
function write_tensor(surf, n, filename, resolution)
    local verts, tris
    if n == 0
        verts = [evaltensor(surf, [u, v])
                 for u in range(0.0, stop=1.0, length=resolution)
                 for v in range(0.0, stop=1.0, length=resolution)]
        tris = []
        for i in 2:resolution, j in 2:resolution
            index = (j - 1) * resolution + i
            push!(tris, [index - resolution - 1, index - resolution, index])
            push!(tris, [index, index - 1, index - resolution - 1])
        end
    else
        poly = regularpoly(n)
        verts = [evaltensor(surf, p) for p in vertices(poly, resolution)]
        tris = triangles(n, resolution)
    end
    writeOBJ(verts, tris, filename)
end

"""
    write_tensor_cnet(surf, filename)

Writes the control structure of a tensor product rational Bézier surface to a mesh file.
"""
function write_tensor_cnet(surf, filename)
    d = size(surf, 1) - 1
    open(filename, "w") do f
        for i in 0:d, j in 0:d
            p = surf[i+1,j+1,1:3] / surf[i+1,j+1,4]
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        for i in 1:d+1, j in 1:d
            index = (i - 1) * (d + 1) + j
            println(f, "l $index $(index+1)")
            index = (j - 1) * (d + 1) + i
            println(f, "l $index $(index+d+1)")
        end
    end
end


# Obj -> STL utility

"""
    obj2stl(filenames, outfile)

Write all triangles in the file list `filenames`
into the file designated by the name `outfile`.

Assumes that the files contain triangles with only
`v` and `f` elements.
"""
function obj2stl(filenames, outfile)
    n = 0
    for filename in filenames
        open(filename) do f
            n += count(line -> split(line)[1] == "f", eachline(f))
        end
    end
    open(outfile, "w") do fout
        print(fout, "Generated by obj2stl" * repeat(' ', 60))
        write(fout, UInt32(n))
        for filename in filenames
            open(filename) do f
                v = []
                for line in eachline(f)
                    parts = split(line)
                    if parts[1] == "v"
                        p = map(s -> parse(Float64, s), parts[2:end])
                        push!(v, p)
                    elseif parts[1] == "f"
                        tri = map(s -> parse(Int, s), parts[2:end])
                        n = normalize!(cross(v[tri[2]] - v[tri[1]], v[tri[3]] - v[tri[1]]))
                        for j in 1:3
                            write(fout, Float32(n[j]))
                        end
                        for i in 1:3, j in 1:3
                            write(fout, Float32(v[tri[i]][j]))
                        end
                        write(fout, UInt16(0))
                    end
                end
            end
        end
    end
end

"""
    write_spatch(surf, filename)

Writes the S-patch `surf` into `filename` in the following format:
- first line: <# of sides> <depth>
- all other lines <index> <control point position>
An example for the latter: 1 3 2 25.0 10.0 -1.0
"""
function write_spatch(surf, filename)
    open(filename, "w") do f
        println(f, "$(surf.n) $(surf.d)")
        for (k, v) in surf.cpts
            for i in k
                print(f, "$i ")
            end
            println(f, join(v, " "))
        end
    end
end

"""
    read_spatch(filename)

Reads an S-patch from the file `filename`.
See `write_spatch` for the file format specification.
"""
function read_spatch(filename)
    read_numbers(lst, numtype) = map(s -> parse(numtype, s), lst)
    open(filename) do f
        n, d = read_numbers(split(readline(f)), Int)
        size = binomial(n + d - 1, d)
        result = SPatch(n, d, Dict())
        for _ in 1:size
            line = split(readline(f))
            index = read_numbers(line[1:n], Int)
            point = read_numbers(line[n+1:end], Float64)
            result.cpts[index] = point
        end
        result
    end
end


# Main function

"""
    spatch_test(name, resolution, [g1_continuity])

Reads a GB patch from the file `name`.gbp, and computes an S-patch
with the same boundary, and optimized inner control points.

When `g1_continuity` is `true` (the default), the resulting S-patch
will have the same normal sweep at its boundaries as the GB patch.

The function outputs several files:
- `name`.sp [the S-patch surface]
- `name`.obj [the S-patch surface mesh]
- `name`-cnet.obj [the full S-patch control net]
- `name`-ribbon.obj [the fixed boundary part of the S-patch control net]
- `name`-bezier-cnet.obj [the original GB ribbons]
"""
function spatch_test(name, resolution, g1_continuity = true)
    ribbons = read_ribbons("$name.gbp")
    surf = g1_continuity ? g1_patch(ribbons) : c0_patch(ribbons)
    optimize_controlnet!(surf, g1 = g1_continuity)
    write_spatch(surf, "$name.sp")
    write_surface(surf, "$name.obj", resolution)
    write_cnet(surf, "$name-cnet.obj")
    write_cnet(surf, "$name-ribbon.obj", g1 = g1_continuity, only_fixed = true)
    g1_continuity && write_bezier_cnet(ribbons, "$name-bezier-cnet.obj")
end

Base.@ccallable function julia_main(ARGS::Vector{String})::Cint
    if length(ARGS) < 2
        println(stderr, "Usage: spatch infile outfile [resolution]")
        return 1
    end
    resolution = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 30
    ribbons = read_ribbons(ARGS[1])
    surf = g1_patch(ribbons)
    optimize_controlnet!(surf)
    write_surface(surf, ARGS[2], resolution)
    return 0
end


# Tensor product conversion

"""
    tosimplex(p)

Returns the barycentric coordinates of the (`n`-dimensional) `p` point
relative to the domain simplex consisting of the origin and the unit
points on the axes, i.e., [1,0,..,0], [0,1,0,..,0], .., [0,..,0,1].
"""
tosimplex(p) = [1 - sum(p); p]

"""
    fromsimplex(bc)

Returns the coordinates of the point with barycentric coordinates `bc`
relative to the domain simplex consisting of the origin and the unit
points on the axes, i.e., [1,0,..,0], [0,1,0,..,0], .., [0,..,0,1].
"""
fromsimplex(bc) = bc[2:end]

"""
    inc_index(index, j, [n])

Returns a new index that is the same as `index`, but with the `j`-th position
increased by `n`.
"""
function inc_index(index, j, n = 1)
    result = copy(index)
    result[j] += n
    result
end

"""
    subindices(index, d)

Returns the list of all valid indices `i` s.t. `index - i` sums to `d`.
"""
subindices(index, d) = filter(i -> all((index - i) .>= 0), indices(length(index), sum(index) - d))

"""
    partitions(index, k)

Returns all partitions of `index` to indices of the same length s.t. all indices sum to `k`,
and the sum of all indices is `index`.
"""
function partitions(index, k)
    result = []
    n = length(index)
    s = sum(index) ÷ k
    function rec(rest, i, current, count, acc)
        if length(acc) == s
            push!(result, acc)
            return
        end
        if count == k
            next = copy(acc)
            push!(next, current)
            return rec(rest, 1, zeros(Int, n), 0, next)
        end
        for j in i:n
            if rest[j] > 0
                rec(inc_index(rest, j, -1), j, inc_index(current, j), count + 1, acc)
            end
        end
    end
    rec(index, 1, zeros(Int, n), 0, [])
    sort!(result)
    unique!(result)
end

"""
    blossom(index, p, V)

Generalized blossoming.
"""
function blossom(index, p, V)
    n = length(p[1])
    length(p) == 1 && return sum(alpha -> p[1][alpha] * V[inc_index(index, alpha)], 1:n)
    sum(alpha -> p[end][alpha] * blossom(inc_index(index, alpha), p[1:end-1], V), 1:n)
end

"""
    compose(S, f)

Computes the composed simplex `S ∘ f`.

As in: T. DeRose, Composing Bézier Simplexes. ACM ToG 7(3), pp. 198-221, 1988.
"""
function compose(S, f)
    # function V(s, i, r) # see Theorem 4.1
    #     s == 0 && return S.cpts[i]
    #     indices = subindices(r, f.d)
    #     isempty(indices) && return zeros(length(iterate(S.cpts)[1][2]))
    #     sum(indices) do j
    #         C = f.cpts[r - j]
    #         W = sum(alpha -> C[alpha] * V(s - 1, inc_index(i, alpha), j), 1:S.n)
    #         multinomial(j) * multinomial(r - j) * W
    #     end / multinomial(r)
    # end
    function V(s, i, r) # see Claim 4.3
        sum(partitions(r, f.d)) do js
            blossom(i, map(j -> f.cpts[j], js), S.cpts) *
                prod(j -> multinomial(j), js) / multinomial(r)
        end
    end
    n = f.n
    d = S.d * f.d
    SPatch(n, d, Dict([r => V(S.d, zeros(Int, S.n), r) for r in indices(n, d)]))
end

"""
    successor!(i)

Destructively changes `i` to the next multi-index in lexicographic order, and returns it.
Returns `nothing` when called with the maximum index.
"""
function successor!(i)
    p = findlast(x -> x != 0, i)
    p == 1 && return nothing
    v = i[p]
    i[p-1] += 1
    i[p] = 0
    i[end] = v - 1
    i
end

"""
    compose_fast(F, G)

Computes the composed simplex `F ∘ G`.

As in: T. DeRose et al., Functional composition algorithms via blossoming.
ACM ToG 12(2), pp. 113-135, 1993.
"""
function compose_fast(F, G)
    n, d = G.n, F.d * G.d
    m = length(iterate(F.cpts)[1][2])
    H = SPatch(n, d, Dict([i => zeros(m) for i in indices(n, d)]))
    Fbar = [Dict() for _ in 1:F.d+1]
    Fbar[1] = Dict([i => p for (i, p) in F.cpts])
    function eval_blossom_arg(p, u)
        for i in indices(F.n, F.d - p)
            Fbar[p+1][i] = sum(j -> Fbar[p][inc_index(i,j)] * u[j], 1:F.n)
        end
    end
    function recursive_compose(k, min, sum, c, mu)
        if k == F.d
            H[sum] += Fbar[F.d+1][zeros(Int,F.n)] * c
        else
            i = copy(min)
            while i != nothing
                eval_blossom_arg(k + 1, G[i])
                recursive_compose(k + 1, i, sum + i, c * multinomial(i) / mu, mu + 1)
                mu = 1
                i = successor!(i)
            end
        end
    end
    recursive_compose(0, make_index(n, (n, G.d)), zeros(Int, n), factorial(F.d), 1)
    for i in keys(H.cpts)
        H[i] /= multinomial(i)
    end
    H
end

"""
    line_point_distance(l1, l2, p)

Returns the signed distance of the point `p` from the line defined
by the two points `l1` and `l2`.
"""
function line_point_distance(l1, l2, p)
    d = p - l1
    t = normalize(l2 - l1)
    norm(d - t * dot(d, t)) * sign(det([p - l1 p - l2]))
end

"""
    permutations(n)

Returns a vector of all permutations of [1..n].
"""
function permutations(n)
    n == 1 && return [[1]]
    vcat([map(i -> insert!(copy(p), i, n), 1:n) for p in permutations(n - 1)]...)
end

"""
    quadify(surf)

Converts an arbitrary S-patch into a 4-sided rational S-patch.
"""
function quadify(surf)
    n = surf.n
    n == 4 && return SPatch(surf.n, surf.d, Dict([i => [p; 1.0] for (i, p) in surf.cpts]))

    # Helper functions
    poly = regularpoly(n)
    dist(i, q) = line_point_distance(poly[i], poly[mod1(i+1,n)], q)
    function polarization(p)
        one(i) = sum(permutations(n - 2)) do perm
            result = 1.0
            k = 1
            for j in 1:n-2
                if k == mod1(i - 1, n) || k == i
                    k = i + 1
                end
                result *= dist(k, p[perm[j]])
                k += 1
            end
            result
        end / factorial(n - 2)
        [one(i) for i in 1:n]
    end
    points(i) = reduce(vcat, [repeat([fromsimplex(s)], j) 
                              for (s, j) in zip([[1.,0,0], [0,1,0], [0,0,1]], i)])

    # Setup the simplexes
    A = SPatch(4, 1, Dict([1,0,0,0] => tosimplex([0.0, 1.0]),
                          [0,1,0,0] => tosimplex([1.0, 1.0]),
                          [0,0,1,0] => tosimplex([1.0, 0.0]),
                          [0,0,0,1] => tosimplex([0.0, 0.0])))
    L = SPatch(3, n - 2, Dict([i => polarization(points(i)) for i in indices(3, n - 2)]))
    B = SPatch(surf.n, surf.d, Dict([i => [p; 1.0 - sum(p)] for (i, p) in surf.cpts]))
    @time S = compose_fast(B, compose_fast(L, A))
    for (i, p) in S.cpts
        S[i][end] = sum(p)   # change between the two kinds of homogeneous coordinates
    end
    S
end

"""
    tensor(surf)

Covnerts a 4-sided rational S-patch into a tensor product rational Bézier patch.
"""
function tensor(surf)
    @assert surf.n == 4 "Only 4-sided S-patches can be directly converted to tensor product form"
    d = surf.d
    result = zeros(d + 1, d + 1, 4)
    for (si, p) in surf.cpts
        i = si[2] + si[3]
        j = si[3] + si[4]
        result[i+1,j+1,:] += p * multinomial(si) / (binomial(d, i) * binomial(d, j))
    end
    result
end

"""
    tensor_test(name, resolution, [g1_continuity])

Reads a GB patch from the file `name`.gbp, and computes an S-patch
with the same boundary, and optimized inner control points.

When `g1_continuity` is `true` (the default), the resulting S-patch
will have the same normal sweep at its boundaries as the GB patch.

Then it is converted into a tensor product Bézier surface.

The function outputs several files:
- `name`-bezier-cnet.obj [the original GB ribbons]
- `name`-cnet.obj [the S-patch control net]
- `name`.obj [the S-patch surface mesh]
- `name`-quad.obj [the quadrilateral S-patch]
- `name`-tensor-cnet.obj [the tensor product control net]
- `name`-tensor.obj [the trimmed tensor product patch]
- `name`-tensor-full.obj [the whole tensor product patch]
"""
function tensor_test(name, resolution, g1_continuity = true)
    ribbons = read_ribbons("$name.gbp")
    println("Hole filling...")
    surf = g1_continuity ? g1_patch(ribbons) : c0_patch(ribbons)
    optimize_controlnet!(surf, g1 = g1_continuity)
    println("Quadifying...")
    quad = quadify(surf)
    println("d = $(quad.d)")
    println("Creating tensor product patch...")
    tsurf = tensor(quad)
    println("Writing meshes...")
    g1_continuity && write_bezier_cnet(ribbons, "$name-bezier-cnet.obj")
    write_cnet(surf, "$name-cnet.obj")
    write_surface(surf, "$name.obj", resolution)
    write_surface_rat(quad, "$name-quad.obj", resolution)
    write_tensor_cnet(tsurf, "$name-tensor-cnet.obj")
    write_tensor(tsurf, surf.n, "$name-tensor.obj", resolution)
    write_tensor(tsurf, 0, "$name-tensor-full.obj", resolution)
end


# Warren's patch

"""
    warrenize!(triangle, m)

Converts all control points to homogenous coordinates, and sets base points based on `m`:
all indices `[i,j,k]` get a weight of 0, when j + k < m[1] or i + k < m[2] or i + j < m[3].
"""
function warrenize!(triangle, m)
    @assert triangle.n == 3 "This function only works on Bezier triangles."
    for (k, v) in triangle.cpts
        if k[2] + k[3] < m[1] || k[1] + k[3] < m[2] || k[1] + k[2] < m[3]
            triangle.cpts[k] = [0, 0, 0, 0]
        elseif length(v) == 3
            triangle.cpts[k] = [v; 1]
        else
            triangle.cpts[k][4] = 1
        end
    end
end

"""
    eval_one_rat(surf, poly, p)

Evaluates the rational S-patch `surf` at domain point `p` inside the domain polygon `poly`.
"""
function eval_one_rat(surf, poly, p)
    result = [0, 0, 0, 0]
    bc = barycentric(poly, p)
    for (i, q) in surf.cpts
        result += q * bernstein(i, bc)
    end
    result[1:3] / result[4]
end

"""
    eval_all_rat(surf, resolution)

Evaluates the rational S-patch `surf` at a given `resolution` (interpreted as in `vertices`),
and returns a pair of arrays (vertices, triangles).
"""
function eval_all_rat(surf, resolution)
    poly = regularpoly(surf.n)
    ([eval_one_rat(surf, poly, p) for p in vertices(poly, resolution)],
     triangles(surf.n, resolution))
end

"""
    write_surface_rat(surf, filename, resolution)

Writes the rational S-patch `surf` into a Wavefront Object file designated by
`filename` with sampling rate `resolution` (interpreted as in `vertices`).
"""
function write_surface_rat(surf, filename, resolution)
    vs, ts = eval_all_rat(surf, resolution)
    writeOBJ(vs, ts, filename)
end

end # module
