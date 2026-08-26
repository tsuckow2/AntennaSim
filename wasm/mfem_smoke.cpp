#include "mfem.hpp"

#include <cmath>

using namespace mfem;

extern "C" {

// Build a tiny 3D H(curl) system and return its true-dof count.
// A negative result identifies the stage that failed.
int mfem_hcurl_smoke() {
  Mesh mesh = Mesh::MakeCartesian3D(1, 1, 1, Element::TETRAHEDRON);
  if (mesh.Dimension() != 3) return -1;

  ND_FECollection fec(1, mesh.Dimension());
  FiniteElementSpace space(&mesh, &fec);
  if (space.GetTrueVSize() <= 0) return -2;

  ConstantCoefficient one(1.0);
  BilinearForm form(&space);
  form.AddDomainIntegrator(new CurlCurlIntegrator(one));
  form.AddDomainIntegrator(new VectorFEMassIntegrator(one));
  form.Assemble();
  form.Finalize();

  const SparseMatrix &matrix = form.SpMat();
  if (matrix.Height() != space.GetVSize() || matrix.Width() != space.GetVSize()) return -3;

  double diagonal_sum = 0.0;
  for (int row = 0; row < matrix.Height(); ++row) {
    const double value = matrix(row, row);
    if (!std::isfinite(value)) return -4;
    diagonal_sum += value;
  }
  if (!std::isfinite(diagonal_sum) || diagonal_sum <= 0.0) return -5;
  return space.GetTrueVSize();
}

}
