import pygsp.graphs as graphs
import pygsp
import numpy as np
import scipy
import scipy.sparse as sp
import sklearn
import matplotlib.pyplot as plt
from tqdm import tqdm

class Sparsifier:
    # Initialize
    # Params: Graph: a pygsp graph
    def __init__(self, G = None, c1=1,c2=1,epsilon=0.5):
        self.G = G
        R,C,V = sp.find(G.W)
        self.R = R; self.C = C; self.V = V;
        n = G.N; self.n = n
        m = len(V); self.m = m
        self.B = sp.csr_matrix((m,n), dtype=np.int8)
        self.W_12 = sp.csr_matrix((m,m))

        for i in range(m):
            r,c,v = R[i], C[i], V[i]
            sgn = (r>c)*1
            self.B[i,r] = sgn; self.B[i,c] = -sgn
            self.W_12[i,i] = np.sqrt(v)
            
        self.k = int(c1 * 24 * np.log(n) / epsilon**2)
        self.k = min(self.k,n)
        
        self.q = int(c2 * n * np.log(n) / epsilon**2)
        self.q = min(self.q,m)
        
        self.ran_sparse = False
        
    def Johnson_Lindenstrauss_Projection(self):
        Q_np = np.random.choice([-1/np.sqrt(self.k), 1/np.sqrt(self.k)],self.k*self.m).reshape(self.k,self.m)
        Q = sp.csr_matrix(Q_np)
        self.Y = Q@self.W_12@self.B
        
    def Find_Z(self):
        
        L = self.G.L
        Z = sp.csr_matrix(self.Y.shape)

        for i in tqdm(range(self.k)):
            y = self.Y.getrow(i)
            z = sp.linalg.spsolve(L,y.T)
            Z[i:,] = z
        
        self.Z = Z
    
    def Approximate_ER(self):
        Re = []
        for i in tqdm(range(self.m)):
            re = sp.linalg.norm(self.Z.getcol(self.R[i])-self.Z.getcol(self.C[i]))**2
            Re.append(re)

        self.Re = Re
    
    def Sample_Edges(self):
        
        Ps = self.Re * self.V; Ps = Ps/np.sum(Ps)
        
        MAX_ITER = 5
        
        for e in range(MAX_ITER):
            W_sparse = sp.csr_matrix((self.n,self.n))
            for s in tqdm(range(self.q)):
                e = np.random.choice(range(self.m),p=Ps)
                w_adj = self.V[e]/(self.q*Ps[e])
                W_sparse[self.R[e],self.C[e]] += w_adj
                W_sparse[self.C[e],self.R[e]] += w_adj;

            self.G_sparse = graphs.Graph(W_sparse)
            if self.G_sparse.is_connected():
                break
        
    def sparsify(self, verbose = False):
        if verbose:
            print(f"Calculating Johnson Lindenstrauss Random Projections With k = {self.k}")
        self.Johnson_Lindenstrauss_Projection()
        if verbose:
            print(f"Solving for the Rows of Z")         
        self.Find_Z()
        if verbose:
            print(f"Calculating Pairwise Distances in JL Space for Effective Resistances")          
        self.Approximate_ER()
        if verbose:
            print(f"Randomly Sampling {self.q} times")      
        self.Sample_Edges()
        self.ran_sparse = True
        
    def get_sparsifier(self):
        if self.ran_sparse:
            try:
                self.G_sparse.coords = self.G.coords
            except:
                None
                
            return self.G_sparse
        else:
            print("Have Not Yet Calculated Sparisfier. Call sparsify()")