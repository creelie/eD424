# Assemble the chunk files written by build_part.jl into one ClusteredLowRankSDP.
function assemble_sdp(files, prec)
    Ablocks = Dict{Any, Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}}(); sizes = Dict{Any, Int}()
    cvals = Dict{Int, CL.Arblib.Arb}()
    m = 0
    mmax = parse(Int, get(ENV, "MMAX", "0"))     # > 0: keep only the constraints 1:mmax (a test sub-programme)
    for f in files
        b = basename(f)
        (mmax > 0 && b[1] == 'k' && b[2] == '4' && parse(Int, b[4:8]) > mmax) && continue
        r = deserialize(f)
        m = mmax > 0 ? mmax : r.m
        for (nm, d) in r.Ablocks
            dd = get!(Ablocks, nm) do; Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}(); end
            for (i, M) in d; (mmax > 0 && i > mmax) && continue; @assert !haskey(dd, i); dd[i] = M; end
        end
        for (nm, sz) in r.sizes; @assert get(sizes, nm, sz) == sz; sizes[nm] = sz; end
        for (i, v) in r.cvals; (mmax > 0 && i > mmax) && continue; @assert !haskey(cvals, i); cvals[i] = v; end
    end
    for nm in collect(keys(Ablocks)); isempty(Ablocks[nm]) && (delete!(Ablocks, nm); delete!(sizes, nm)); end
    @assert sort(collect(keys(cvals))) == collect(1:m) "missing samples: have $(length(cvals)) of $m"
    names = collect(keys(Ablocks))
    A = [[begin
            M = Matrix{Dict{Int, Union{CL.LowRankMat, CL.ArbRefMatrix}}}(undef, 1, 1); M[1, 1] = Ablocks[nm]; M
          end for nm in names]]
    Cb = [CL.ArbRefMatrix(sizes[nm], sizes[nm], prec=prec) for nm in names]
    for (l, nm) in enumerate(names)
        nm == [0, 0] && (Cb[l][1, 1] = 1)
    end
    C = CL.BlockDiagonal([CL.BlockDiagonal(Cb)])
    c = CL.ArbRefMatrix(m, 1, prec=prec)
    for i = 1:m; c[i, 1] = cvals[i]; end
    B = [CL.ArbRefMatrix(m, 0, prec=prec)]
    b = CL.ArbRefMatrix(0, 1, prec=prec)
    return CL.ClusteredLowRankSDP(false, CL.Arblib.Arb(0, prec=prec), A, B, [c], C, b, [names], Any[], [[(false, 1) for _ in names]])
end
