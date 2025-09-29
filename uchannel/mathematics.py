import math
from typing import Tuple
import numpy as np
import gmsh

def unit(v):
    n = math.sqrt(v[0]*v[0]+v[1]*v[1]+v[2]*v[2])
    return (v[0]/n, v[1]/n, v[2]/n) if n>1e-15 else (0.0,0.0,0.0)

def sub(a,b): return a[0]-b[0], a[1]-b[1], a[2]-b[2]

def add(a,b): return a[0]+b[0], a[1]+b[1], a[2]+b[2]

def smul(s,v): return s*v[0], s*v[1], s*v[2]

def dot(a,b): return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def cross(a,b): return a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0]

def clamp(x,a,b): return a if x<a else (b if x>b else x)

def edge_endpoints(edge:int) -> tuple:
    u0,u1 = gmsh.model.getParametrizationBounds(1, edge)
    p0 = tuple(gmsh.model.getValue(1, edge, [float(u0)]))
    p1 = tuple(gmsh.model.getValue(1, edge, [float(u1)]))
    return p0, p1

def tangent_from_corner(edge:int, p:Tuple[float,float,float]) -> Tuple[float,float,float]:
    u0,u1 = gmsh.model.getParametrizationBounds(1, edge)
    p0 = tuple(gmsh.model.getValue(1, edge, [float(u0)]))
    p1 = tuple(gmsh.model.getValue(1, edge, [float(u1)]))
    d0 = math.dist(p0, p); d1 = math.dist(p1, p)
    u_corner, step = (u0, +1.0) if d0 <= d1 else (u1, -1.0)
    du = (u1-u0) if u1!=u0 else 1.0
    us = u_corner + step*max(abs(du)*1e-6, 1e-9)
    q = tuple(gmsh.model.getValue(1, edge, [float(us)]))
    return unit(sub(q, p))

def distance(p1, p2):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))

def split_by_centerline(graph, nodes, p1, p2):
    """Split nodes into 'above' and 'below' relative to a plane defined by a centerline."""
    p1 = np.array(p1, dtype=float)
    p2 = np.array(p2, dtype=float)

    # Direction of the centerline
    axis_vec = p2 - p1
    axis_vec /= np.linalg.norm(axis_vec)

    # Reference vector: global Z (or fallback if parallel)
    ref = np.array([0,0,1], dtype=float)
    if np.allclose(np.abs(np.dot(axis_vec, ref)), 1.0, atol=1e-6):
        ref = np.array([0,1,0], dtype=float)

    # Plane normal = perpendicular to axis and reference
    normal = np.cross(axis_vec, ref)
    normal /= np.linalg.norm(normal)

    # Plane equation through p1
    d = -np.dot(normal, p1)

    below, above = [], []
    for node in nodes:
        coord = np.array(graph.nodes[node]["coords"], dtype=float)
        side = np.dot(normal, coord) + d
        if side < 0:
            below.append(node)
        else:
            above.append(node)

    return below, above