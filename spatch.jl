using LinearAlgebra

indices(n, d) = n == 1 ? [d] : [[i; xs] for i in 0:d for xs in indices(n - 1, d - i)]

function isboundary(xs)
    n = length(xs)
    d = sum(xs)
    any(i -> xs[i] + xs[mod1(i + 1, n)] == d, 1:n)
end

function neighbors(xs)
    n = length(xs)
    function step(i, dir)
        j = mod1(i + dir, n)
        [k == i ? x - 1 : (k == j ? x + 1 : x) for (k, x) in enumerate(xs)]
    end
    valid(ys) = all(y -> y >= 0, ys)
    filter(valid, [[step(i, 1) for i in 1:n]; [step(i, -1) for i in 1:n]])
end

isribbon(xs) = any(isboundary, neighbors(xs))

function harmonic_mask(xs)
    neigs = neighbors(xs)
    result = Dict(xs => -length(neigs))
    foreach(neig -> result[neig] = 1, neigs)
    result
end

function biharmonic_mask(xs)
    result = Dict{Vector{Int},Int}()
    for (ys, weight) in harmonic_mask(xs)
        for (p, w) in harmonic_mask(ys)
            result[p] = get!(result, p, 0) + w * weight
        end
    end
    result
end

function optimize_controlnet!(cnet; g1 = true)
    isfixed = g1 ? isribbon : isboundary
    n = length(first(keys(cnet)))
    d = sum(first(keys(cnet)))
    movable = filter(i -> !isfixed(i), indices(n, d))
    mapping = Dict(map(p -> (p[2], p[1]), enumerate(movable)))
    m = length(movable)
    A = zeros(m, m)     # or maybe A should be n x m and use LSQ?
    b = zeros(m, 3)
    for (i, p) in enumerate(movable)
        for (q, w) in biharmonic_mask(p)
            if isfixed(q)
                b[i,1:end] -= cnet[q] * w
            else
                A[i,mapping[q]] = w
            end
        end
    end
    x = A \ b
    for (i, p) in enumerate(movable)
        cnet[p] = x[i,1:end]
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

function eval_one(poly, cnet, p)
    result = [0, 0, 0]
    bc = barycentric(poly, p)
    for (i, q) in cnet
        result += q * bernstein(i, bc)
    end
    result
end

affine_combine(p, x, q) = p * (1 - x) + q * x

function vertices(poly, resolution)
    n = length(poly)
    lines = [(poly[mod1(i - 1, n)], poly[i]) for i in 1:n]
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

function eval_all(cnet, resolution)
    n = length(first(keys(cnet)))
    poly = regularpoly(n)
    ([eval_one(poly, cnet, p) for p in vertices(poly, resolution)],
     triangles(n, resolution))
end


# Input

function boundary_indices(n, d)
    function step!(xs)
        i = 1
        while xs[i] == 0 || xs[i] + xs[mod1(i + 1, n)] != d
            i += 1
        end
        xs[i] -= 1
        xs[mod1(i + 1, n)] += 1
    end
    xs = [d; zeros(Int, n - 1)]
    result = []
    for i in 1:n*d
        push!(result, copy(xs))
        step!(xs)
    end
    result
end

function ribbon_indices(n, d)
    @assert d > 2 "This function does not work for degree <= 2"
    function step(xs)
        i = 1
        while xs[i] == 0 || xs[i] + xs[mod1(i + 1, n)] != d
            i += 1
        end
        xs[i] == d && return nothing
        right = copy(xs)
        right[mod1(i + 2, n)] += 1
        right[mod1(i + 1, n)] -= 1
        xs[i] == d - 1 && return right
        left = copy(xs)
        left[mod1(i - 1, n)] += 1
        left[i] -= 1
        (left, right)
    end
    boundary = boundary_indices(n, d)
    points = filter(x -> x != nothing, [step(xs) for xs in boundary])
    result = [[]]
    for p in points
        if isa(p, Tuple)
            push!(result, [p[1], p[2]])
        else
            push!(result[end], p)
        end
    end
    push!(result[1], result[end]...)
    result[1:end-1]
end

function readGBP(filename)
    read_numbers(f, numtype) = map(s -> parse(numtype, s), split(readline(f)))
    result = Dict{Vector{Int},Vector{Float64}}()
    open(filename) do f
        n, d = read_numbers(f, Int)
        readline(f)
        for i in boundary_indices(n, d)
            result[i] = read_numbers(f, Float64)
        end
        for idx in ribbon_indices(n, d)
            p = read_numbers(f, Float64)
            foreach(i -> result[i] = p, idx)
        end
    end
    result
end


# Output

function write_cnet(cnet, filename; g1 = true, only_fixed = false)
    isfixed = g1 ? isribbon : isboundary
    mapping = Dict(map(p -> (p[2], p[1]), enumerate(keys(cnet))))
    open(filename, "w") do f
        for p in values(cnet)
            println(f, "v $(p[1]) $(p[2]) $(p[3])")
        end
        for i in keys(cnet)
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

function write_surface(cnet, filename, resolution)
    vs, ts = eval_all(cnet, resolution)
    writeOBJ(vs, ts, filename)
end

function spatch_test(name, resolution, g1_continuity = true)
    cnet = readGBP("$name.gbp")
    optimize_controlnet!(cnet, g1 = g1_continuity)
    write_surface(cnet, "$name.obj", resolution)
    write_cnet(cnet, "$name-cnet.obj")
    write_cnet(cnet, "$name-ribbon.obj", g1 = g1_continuity, only_fixed = true)
end


# Bezier ribbon evaluation for comparison

function read_ribbons(filename)
    read_numbers(f, numtype) = map(s -> parse(numtype, s), split(readline(f)))
    result = Dict{Tuple{Int,Int,Int},Vector{Float64}}()
    open(filename) do f
        n, d = read_numbers(f, Int)
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
            result[(side,col,row)] = p
            if col < l
                result[(mod(side-1,n),d-row,col)] = p
            elseif d - col < l
                result[(mod(side+1,n),row,d-col)] = p
            end
            col += 1
        end
    end
    result
end

function write_ribbon(ribbons, filename, index, resolution)
    d = maximum(map(x -> x[2], collect(keys(ribbons))))
    bezier(n, k, x) = binomial(n, k) * x ^ k * (1 - x) ^ (n - k)
    samples = [[u, v] for u in range(0.0, stop=1.0, length=resolution)
                      for v in range(0.0, stop=1.0, length=resolution)]
    verts = map(samples) do
        result = [0, 0, 0]
        for i in 0:d, j in 0:1
            result += ribbons[(index,i,j)] * bezier(d, i, p[1]) * bezier(1, j, p[2])
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
