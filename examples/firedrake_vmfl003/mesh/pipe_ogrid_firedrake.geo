// VMFL003 five-block O-grid cross-section, generated in 2-D and extruded by Firedrake.
// Physical Curve 1 = wall, Physical Surface 10 = fluid.

SetFactory("Built-in");

DefineConstant[
  R = {0.002, Name "Geometry/Radius"},
  CoreRatio = {0.50, Name "Mesh/Core half-width / Radius"},
  Nq = {12, Min 2, Max 100, Step 1, Name "Mesh/Cells per quarter-circle"},
  Nr = {1, Min 1, Max 100, Step 1, Name "Mesh/Radial cells in O-grid"}
];

a = CoreRatio * R;
c = R / Sqrt(2);

Point(1) = {-a, -a, 0};
Point(2) = { a, -a, 0};
Point(3) = { a,  a, 0};
Point(4) = {-a,  a, 0};

Point(5) = { c, -c, 0};
Point(6) = { c,  c, 0};
Point(7) = {-c,  c, 0};
Point(8) = {-c, -c, 0};
Point(9) = {0, 0, 0};

Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

Line(5) = {1,8};
Line(6) = {2,5};
Line(7) = {3,6};
Line(8) = {4,7};

Circle(9)  = {5,9,6};
Circle(10) = {6,9,7};
Circle(11) = {7,9,8};
Circle(12) = {8,9,5};

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

Curve Loop(2) = {-2,6,9,-7};
Plane Surface(2) = {2};

Curve Loop(3) = {-3,7,10,-8};
Plane Surface(3) = {3};

Curve Loop(4) = {-4,8,11,-5};
Plane Surface(4) = {4};

Curve Loop(5) = {-1,5,12,-6};
Plane Surface(5) = {5};

Transfinite Curve {1,2,3,4,9,10,11,12} = Nq + 1;
Transfinite Curve {5,6,7,8} = Nr + 1;

Transfinite Surface {1};
Transfinite Surface {2};
Transfinite Surface {3};
Transfinite Surface {4};
Transfinite Surface {5};

Recombine Surface {1,2,3,4,5};
Mesh.Smoothing = 20;
Mesh.ElementOrder = 1;
Mesh.Binary = 0;

// Firedrake boundary/subdomain labels.
Physical Curve("wall", 1) = {9,10,11,12};
Physical Surface("fluid", 10) = {1,2,3,4,5};
