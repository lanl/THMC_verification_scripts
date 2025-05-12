# THMC_verification_scripts
Verification scripts associated with "Analytic Solutions and Field-Scale Application for Verification of Coupled Thermo-Hydro-Mechanical Processes in Subsurface Fractured Media" By Huang et al. 

# Benchmark Problems for Hydro-Mechanical and Thermo-Mechanical Models

This repository contains benchmark problems designed to validate and compare numerical simulations with analytical or established numerical solutions in the context of hydro-mechanical (HM) and thermo-mechanical (TM) processes.

---

## Benchmark Cases

### 1. Lauwerier (TH)
**Type:** Thermo-Hydraulic (TH)  
**Description:** Simulates the injection of water into a single fracture embedded in a thermally active rock matrix.  
**Validation:** Compares numerical results with the analytical solution for temperature distribution in both the fracture and the surrounding matrix.

---

### 2. Terzaghi (HM)
**Type:** Hydro-Mechanical (HM)  
**Description:** Models the drainage of a 1D porous medium subjected to a constant stress applied at one boundary. Drainage is permitted only at the loaded boundary.  
**Validation:** Compared against the analytical pressure solution.

---

### 3. Schiffman (HM)
**Type:** Hydro-Mechanical (HM)  
**Description:** Extension of the Terzaghi problem with a linearly increasing boundary pressure instead of a constant one.  
**Validation:** Compared against the analytical pressure solution.

---

### 4. Mandel (HM)
**Type:** Hydro-Mechanical (HM)  
**Description:** Simulates the compression of a 2D porous medium confined between two impermeable plates. Drainage is allowed only at the open boundary.  
**Validation:** Compared against the analytical solution for pressure evolution.

---

### 5. Wijesinghe (HM)
**Type:** Hydro-Mechanical (HM)  
**Description:** Models fluid injection into a single fracture from one end, with drainage allowed at the opposite end. The surrounding matrix is impermeable.  
**Validation:** Compared against a numerical solution derived from a set of ODEs describing pressure in the fracture.

---

### 6. McNamee-Gibons (HM)
**Type:** Hydro-Mechanical (HM)  
**Description:** Applies a face load to a strip on the surface of the domain.  
**Validation:** Compared with multiphysics COMSOL simulation results. COMSOL results are benchmarked against analytical solutions. This indirect comparison is due to current issues handling incompressible fluids in the in-house code.

---

### 7. Noda_TM (TM) *(In Development)*
**Type:** Thermo-Mechanical (TM)  
**Description:** Models a hollow cylinder with different temperature conditions applied to the inner and outer surfaces.  
**Validation:** Will be compared with analytical solutions for both temperature distribution and induced stresses.

---

## License

This program is Open-Source under the BSD-3 License.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

- Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
- Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
- Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

**DISCLAIMER:**  
THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.  
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES  
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND  
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,  
EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.