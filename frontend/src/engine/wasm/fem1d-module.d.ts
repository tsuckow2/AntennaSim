export interface Fem1dModule {
  _malloc(size: number): number;
  _free(pointer: number): void;
  _fem_solve(
    lengthM: number,
    startMhz: number,
    stopMhz: number,
    steps: number,
    meshSizeM: number,
    referenceImpedanceOhms: number,
    outputPointer: number,
  ): number;
  HEAPF64: Float64Array;
}
