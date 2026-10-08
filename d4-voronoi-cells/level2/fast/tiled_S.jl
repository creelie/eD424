# Tiled Schur-complement assembly for ClusteredLowRankSolver 1.0.3 (evaluated into its module).
# The original compute_S_integrated! forms, for each low-rank block, the full matrices
# V^T Y V and V^T X^{-1} V of size (#samples * rank)^2 -- 2 x 5.85 GB at (14,16) in 128-bit
# Arb -- and solvesdp preallocates them.  Here the samples are cut into tiles, only the
# tile pairs (a <= b) of the upper triangle are formed (half the multiplications), and the
# buffers are tile-sized.  Requires what holds for the LLM24 programme: every low-rank
# block has a single subblock and symmetric low-rank matrices (vs == ws).
const TILE = Ref(1000)          # samples per tile
const TILEBUF = Dict{Symbol, Any}()
const GRAM_HIGHRANK = Ref(false)

function tilebuf(sym, r, c, prec)
    M = get(TILEBUF, sym, nothing)
    if M === nothing || size(M, 1) < r || size(M, 2) < c || precision(M) != prec
        TILEBUF[sym] = nothing; GC.gc()
        M = ArbRefMatrix(r, c, prec=prec)
        TILEBUF[sym] = M
    end
    return M
end

function compute_S_integrated_tiled!(S,sdp,A_Y, X_inv, Y,bilinear_pairings_Y, bilinear_pairings_Xinv, leftvecs,rightvecs,pointers_left,pointers_right,high_ranks, tempX, temppart_r;matmul_prec=precision(S[1]))
    prec = precision(Y)
    w1 = ArbRefMatrix(0,0, prec=matmul_prec); w2 = ArbRefMatrix(0,0, prec=matmul_prec)
    w3 = ArbRefMatrix(0,0, prec=matmul_prec); w4 = ArbRefMatrix(0,0, prec=matmul_prec)
    w5 = ArbRefMatrix(0,0, prec=matmul_prec); w6 = ArbRefMatrix(0,0, prec=matmul_prec)
    for j in eachindex(sdp.A)
        Arblib.zero!(S[j])
        for l in eachindex(sdp.A[j])
            if high_ranks[j][l] && !GRAM_HIGHRANK[]
                # unchanged from ClusteredLowRankSolver 1.0.3
                for p in keys(sdp.A[j][l][1,1])
                    Arblib.solve_cho_precomp!(tempX[1].blocks[j].blocks[l], X_inv.blocks[j].blocks[l], sdp.A[j][l][1,1][p])
                    matmul_threaded!(tempX[2].blocks[j].blocks[l], tempX[1].blocks[j].blocks[l], Y.blocks[j].blocks[l], prec=matmul_prec)
                    qs = [q for q in keys(sdp.A[j][l][1,1]) if q >= p]
                    Threads.@threads for q in qs
                        Arblib.add!(S[j][p,q],S[j][p,q], dot(sdp.A[j][l][1,1][q],tempX[2].blocks[j].blocks[l]))
                    end
                end
                continue
            elseif high_ranks[j][l]
                # <A_p, X^-1 A_q Y> = <G_p, G_q> with G_p = L_X^-1 A_p L_Y  (X = L_X L_X^T, Y = L_Y L_Y^T):
                # one Gram matrix G G^T, formed by block matrix products, instead of m^2/2 scalar loops
                sz = size(Y.blocks[j].blocks[l], 1)
                LY = ArbRefMatrix(sz, sz, prec=prec)
                Arblib.set!(LY, Y.blocks[j].blocks[l])
                approx_cholesky!(LY); Arblib.get_mid!(LY, LY)
                for i = 1:sz, k = i+1:sz; Arblib.zero!(LY[i, k]); end     # keep the lower factor only
                ps = sort(collect(keys(sdp.A[j][l][1,1])))
                G = ArbRefMatrix(length(ps), sz * sz, prec=matmul_prec)
                Threads.@threads for ip in eachindex(ps)
                    T1 = ArbRefMatrix(sz, sz, prec=prec); T2 = ArbRefMatrix(sz, sz, prec=prec)
                    Arblib.approx_solve_tril!(T1, X_inv.blocks[j].blocks[l], sdp.A[j][l][1,1][ps[ip]], 0)
                    Arblib.approx_mul!(T2, T1, LY)
                    for a = 1:sz, b = 1:sz
                        Arblib.set!(G[ip, (b - 1) * sz + a], T2[a, b])
                    end
                end
                Arblib.get_mid!(G, G)
                Gt = transpose(G)
                np = length(ps); rt = 512
                for r0 = 1:rt:np
                    r1 = min(np, r0 + rt - 1)
                    Gr = ArbRefMatrix(r1 - r0 + 1, sz * sz, prec=matmul_prec)
                    Arblib.window_init!(w1, G, r0 - 1, 0, r1, sz * sz); Arblib.set!(Gr, w1); Arblib.window_clear!(w1)
                    Gram = ArbRefMatrix(r1 - r0 + 1, np, prec=matmul_prec)
                    matmul_threaded!(Gram, Gr, Gt, prec=matmul_prec)
                    Threads.@threads for a = r0:r1
                        p = ps[a]
                        for b = a:np
                            Arblib.add!(S[j][p, ps[b]], S[j][p, ps[b]], Gram[a - r0 + 1, b])
                        end
                    end
                end
                continue
            end
            @assert size(sdp.A[j][l]) == (1,1)
            Adict = sdp.A[j][l][1,1]
            n = size(Y.blocks[j].blocks[l], 1)
            Arblib.inv_cho_precomp!(tempX[1].blocks[j].blocks[l], X_inv.blocks[j].blocks[l])
            Arblib.get_mid!(tempX[1].blocks[j].blocks[l], tempX[1].blocks[j].blocks[l])
            Xi = tempX[1].blocks[j].blocks[l]; Yl = Y.blocks[j].blocks[l]
            kp = collect(keys(Adict))                     # the order A_Y uses
            offs = Dict{Int,Int}(); idx = 0
            for p in kp; offs[p] = idx; idx += length(Adict[p].lambda); end
            tiles = collect(Iterators.partition(sort(kp), TILE[]))
            R = rightvecs[j][l][1]; Lv = leftvecs[j][l][1]
            ptrR = pointers_right[j][l][1]; ptrL = pointers_left[j][l][1]
            tcols = [Int[] for _ in tiles]; trows = [Int[] for _ in tiles]; tloc = [Dict{Int,Int}() for _ in tiles]
            for (ti, tile) in enumerate(tiles), p in tile
                tloc[ti][p] = length(tcols[ti])
                for rnk = 1:length(Adict[p].lambda)
                    push!(tcols[ti], ptrR[(1,p,rnk)]); push!(trows[ti], ptrL[(1,p,rnk)])
                end
            end
            maxc = maximum(length.(tcols))
            VR = tilebuf(:VR, n, maxc, matmul_prec); VL = tilebuf(:VL, maxc, n, matmul_prec)
            PY = tilebuf(:PY, n, maxc, matmul_prec); PX = tilebuf(:PX, n, maxc, matmul_prec)
            BPY = tilebuf(:BPY, maxc, maxc, matmul_prec); BPX = tilebuf(:BPX, maxc, maxc, matmul_prec)
            for b in eachindex(tiles)
                cb = length(tcols[b])
                for (k, c) in enumerate(tcols[b]), i = 1:n
                    Arblib.set!(VR[i, k], R[i, c])
                end
                Arblib.window_init!(w1, VR, 0, 0, n, cb)
                Arblib.window_init!(w2, PY, 0, 0, n, cb)
                Arblib.window_init!(w3, PX, 0, 0, n, cb)
                matmul_threaded!(w2, Yl, w1, prec=matmul_prec)
                matmul_threaded!(w3, Xi, w1, prec=matmul_prec)
                Arblib.get_mid!(w2, w2); Arblib.get_mid!(w3, w3)
                for a = 1:b
                    ca = length(tcols[a])
                    for (k, c) in enumerate(trows[a]), i = 1:n
                        Arblib.set!(VL[k, i], Lv[c, i])
                    end
                    Arblib.window_init!(w4, VL, 0, 0, ca, n)
                    Arblib.window_init!(w5, BPY, 0, 0, ca, cb)
                    Arblib.window_init!(w6, BPX, 0, 0, ca, cb)
                    matmul_threaded!(w5, w4, w2, prec=matmul_prec)
                    matmul_threaded!(w6, w4, w3, prec=matmul_prec)
                    Arblib.get_mid!(w5, w5); Arblib.get_mid!(w6, w6)
                    Arblib.window_clear!(w4); Arblib.window_clear!(w5); Arblib.window_clear!(w6)
                    if a == b
                        for p in tiles[a], rnk = 1:length(Adict[p].lambda)
                            ii = tloc[a][p] + rnk
                            A_Y[j][l][1,1][offs[p] + rnk, 1] = BPY[ii, ii]
                        end
                    end
                    tp = collect(tiles[a]); tq = collect(tiles[b])
                    Threads.@threads for p in tp
                        prodv = AL.Arb(prec=prec)
                        lp = Adict[p].lambda; ip = tloc[a][p]
                        for q in tq
                            (a == b && q < p) && continue
                            lq = Adict[q].lambda; iq = tloc[b][q]
                            for r1 in eachindex(lp), r2 in eachindex(lq)
                                Arblib.mul!(prodv, BPX[ip + r1, iq + r2], BPY[ip + r1, iq + r2])
                                if !(isone(lp[r1]) && isone(lq[r2]))
                                    Arblib.mul!(prodv, prodv, lp[r1]); Arblib.mul!(prodv, prodv, lq[r2])
                                end
                                Arblib.add!(S[j][p, q], S[j][p, q], prodv)
                            end
                        end
                    end
                end
                Arblib.window_clear!(w1); Arblib.window_clear!(w2); Arblib.window_clear!(w3)
            end
        end
        symmetric!(S[j])
        Arblib.get_mid!(S[j], S[j])
    end
    GC.gc(false)
end
