/* Copyright 2024 National Technology & Engineering Solutions of Sandia, LLC (NTESS).
 * Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains
 * certain rights in this software.
 *
 * Written by Jed A. Duersch, Sandia National Laboratories, Livermore, CA.
 *
 * This algorithm implements methods described in the paper, "Curvature in the Looking-Glass:
 * Optimal Methods to Exploit Curvature of Expectation in the Loss Landscape."
 */

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#define THREADS_PER_SM 128

// Indices for matrix of parameter states:
#define MMu 0 // parameter center
#define MNu 1 // look-ahead evaluation
#define MZ 2  // temporary: perturbation at step start, average gradient during steps
#define MG 3  // average gradient
#define MR 4  // average gradient variance density
#define MH 5  // average hessian
#define MV 6  // average temportal variance
#define MLen 7

// Indices for shared floats:
#define FSig 0 // perturbation distance and adam learning rate
#define FBet1 1 // beta in average gradient
#define FBet2 2 // beta in average second moment
#define FBet3 3 // beta2 adjusted for quick-step period
#define FEps 4 // epsilon in denominator.
#define FW1 5 // weight of L1 regularization
#define FW2 6 // weight of L2 regularization
#define FPhi 7 // fraction of quasi-Newton step, mu = mu + phi * delta
#define FOmg 8 // Nesterove accelerated gradient evaluation, nu = mu + omega * delta
#define FTau1 9 // Minimum step scaling factor
#define FTau2 10 // Maximum step scaling factor
#define FLen 11

// Indices for shared ints:
#define IQ 0 // index of perturbation, {0, 1, 2}
#define IH 1 // hessian computation type, {0: 0, 1: abs, 2: rms}
#define IR 2 // include rho in steps
#define IL 3 // limiter method for tau1 and tau2.
#define ILen 4

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CPU(x) TORCH_CHECK(x.device().is_cpu(), #x " must be on CPU")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_CUDA_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)
#define CHECK_CPU_INPUT(x) CHECK_CPU(x);



// =================================
// ========== Init Perturb =========
// =================================
// z < 0.5 : p = nu - sigma
// else    : p = nu + sigma
template <typename scalar_t>
__global__ void alice_init_pert_kernel(
   torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> P,
   torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> M,
   const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> F,
   const int numel) {
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numel) {
      // Alg. 2 Step 1. Convert to Rademacher samples.
      const scalar_t delta = (M[i][MZ] < 0.5 ? -F[FSig] : F[FSig]);
      // Set positive perturbation and store negative perturbation.
      // Alg. 2, Steps 1 - 4.
      P[i] = M[i][MNu] + delta;
      M[i][MZ] = M[i][MNu] - delta;
   }
}

template <typename scalar_t>
void alice_init_pert_cpu(
   torch::TensorAccessor<scalar_t, 1> P,
   torch::TensorAccessor<scalar_t, 2> M,
   const torch::TensorAccessor<scalar_t, 1> F,
   const int numel) {
   for (int i = 0; i < numel; i++) {
      const scalar_t delta = (M[i][MZ] < 0.5 ? -F[FSig] : F[FSig]);
      // Set positive perturbation and store negative perturbation.
      P[i] = M[i][MNu] + delta;
      M[i][MZ] = M[i][MNu] - delta;
   }
}

void init_pert(
   torch::Tensor P,
   torch::Tensor M,
   const torch::Tensor F) {
   const int numel = M.size(0);
   assert(P.numel() == numel);
   assert(F.numel() == FLen);

   torch::Tensor P_ = P.flatten();

   if (P_.device().is_cuda()) {
      CHECK_CONTIGUOUS(P_);
      CHECK_CUDA_INPUT(M);
      CHECK_CUDA_INPUT(F);

      const int threads = THREADS_PER_SM;
      const dim3 blocks((numel + threads - 1) / threads);
      
      AT_DISPATCH_FLOATING_TYPES(
         P_.scalar_type(), "init_pert_cuda",
         ([&] {
            alice_init_pert_kernel<scalar_t><<<blocks, threads>>>(
               P_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               M.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
               F.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               numel);
         }));
   } else {
      CHECK_CPU_INPUT(P_);
      CHECK_CPU_INPUT(M);
      CHECK_CPU_INPUT(F);

      AT_DISPATCH_FLOATING_TYPES(
         P_.scalar_type(), "init_pert_cpu",
         ([&] {
            alice_init_pert_cpu<scalar_t>(
               P_.accessor<scalar_t, 1>(),
               M.accessor<scalar_t, 2>(),
               F.accessor<scalar_t, 1>(),
               numel);
         }));
   }
}

// ========================================
// ========== Perturbation Update =========
// ========================================
template <typename scalar_t>
__global__ void alice_pert_update_kernel(
   torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> P,
   torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> M,
   const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> G,
   const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> F,
   const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> I,
   const int numel) {
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numel) {
      // This only needs to run on active parameters.
      if (I[IQ] == 0) {
         // Set next evaluation location and then save positive perturbation gradient.
         // Alg 2, Step 3.
         P[i] = M[i][MZ];
         M[i][MZ] = G[i];
      } else if (I[IQ] == 1) {
         // Alg 2, Step 7.
         // Set the final evalation location at the look-ahead center.
         P[i] = M[i][MNu];
         // Alg 2, Steps 5 and 6.
         // Use negative gradient to update Hessian and then save the average.
         const scalar_t gp = M[i][MZ];
         const scalar_t gm = G[i];
         scalar_t h = 0.5 * (gp - gm) / F[FSig];
         if (I[IH] == 2) { // Save running hessian squared.
            h = h * h + F[FW2] * F[FW2];
         } else { // Save running absolute value.
            h = abs(h) + F[FW2];
         }
         M[i][MH] = h + F[FBet3] * (M[i][MH] - h);
         // Alg 2, Step 7.
         M[i][MZ] = 0.5 * (gp + gm);
      } else if (I[IQ] == 2) {
         // Use the centered gradient to update average gradient and compute variation density.
         // Alg 2, Steps 9 - 11.
         scalar_t g0 = G[i];
         const scalar_t dg = M[i][MZ] - g0;
         const scalar_t rho = 2.0 * dg * dg / F[FSig];
         const scalar_t v = g0 * g0;
         // Add regularization terms after second moment.
         g0 += (M[i][MNu] < 0 ? -F[FW1] : F[FW1]) + F[FW2] * M[i][MNu];
         M[i][MR] = rho + F[FBet3] * (M[i][MR] - rho);
         // This variable is called s in the paper:
         M[i][MV] = v   + F[FBet2] * (M[i][MV] - v);
         M[i][MG] = g0  + F[FBet1] * (M[i][MG] - g0);
      }
   }
}

template <typename scalar_t>
void alice_pert_update_cpu(
   torch::TensorAccessor<scalar_t, 1> P,
   torch::TensorAccessor<scalar_t, 2> M,
   const torch::TensorAccessor<scalar_t, 1> G,
   const torch::TensorAccessor<scalar_t, 1> F,
   const torch::TensorAccessor<int, 1> I,
   const int numel) {
   for (int i = 0; i < numel; i++) {
      // This only needs to run on active parameters.
      if (I[IQ] == 0) {
         // Set next evaluation location and then save positive perturbation gradient.
         P[i] = M[i][MZ];
         M[i][MZ] = G[i];
      } else if (I[IQ] == 1) {
         // Set the final evalation location at the look-ahead center.
         P[i] = M[i][MNu];
         // Use negative gradient to update Hessian and then save the average.
         const scalar_t gp = M[i][MZ];
         const scalar_t gm = G[i];
         scalar_t h = 0.5 * (gp - gm) / F[FSig];
         if (I[IH] == 2) { // Save running hessian squared.
            h = h * h + F[FW2] * F[FW2];
         } else { // Save running absolute value.
            h = abs(h) + F[FW2];
         }
         M[i][MH] = h + F[FBet3] * (M[i][MH] - h);
         M[i][MZ] = 0.5 * (gp + gm);
      } else if (I[IQ] == 2) {
         // Use the centered gradient to update average gradient and compute variation density.
         scalar_t g0 = G[i];
         const scalar_t dg = M[i][MZ] - g0;
         const scalar_t rho = 2.0 * dg * dg / F[FSig];
         const scalar_t v = g0 * g0;
         // Add regularization terms after second moment.
         g0 += (M[i][MNu] < 0 ? -F[FW1] : F[FW1]) + F[FW2] * M[i][MNu];
         M[i][MR] = rho + F[FBet3] * (M[i][MR] - rho);
         M[i][MV] = v   + F[FBet2] * (M[i][MV] - v);
         M[i][MG] = g0  + F[FBet1] * (M[i][MG] - g0);
      }
   }
}

void pert_update(
   torch::Tensor P,
   torch::Tensor M,
   const torch::Tensor G,
   const torch::Tensor F,
   const torch::Tensor I) {
   const int numel = M.size(0);
   assert(G.numel() == numel);
   assert(F.numel() == FLen);
   assert(I.numel() == ILen);

   torch::Tensor P_ = P.flatten();
   const torch::Tensor G_ = G.flatten();

   if (P_.device().is_cuda()) {
      CHECK_CONTIGUOUS(P_);
      CHECK_CUDA_INPUT(M);
      CHECK_CUDA_INPUT(G_);
      CHECK_CUDA_INPUT(F);
      CHECK_CUDA_INPUT(I);

      const int threads = THREADS_PER_SM;
      const dim3 blocks((numel + threads - 1) / threads);
      
      AT_DISPATCH_FLOATING_TYPES(
         M.scalar_type(), "pert_update_cuda",
         ([&] {
            alice_pert_update_kernel<scalar_t><<<blocks, threads>>>(
               P_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               M.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
               G_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               F.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               I.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
               numel);
         }));
   } else {
      CHECK_CPU_INPUT(P_);
      CHECK_CPU_INPUT(M);
      CHECK_CPU_INPUT(G_);
      CHECK_CPU_INPUT(F);
      CHECK_CPU_INPUT(I);

      AT_DISPATCH_FLOATING_TYPES(
         M.scalar_type(), "pert_update_cpu",
         ([&] {
            alice_pert_update_cpu<scalar_t>(
               P_.accessor<scalar_t, 1>(),
               M.accessor<scalar_t, 2>(),
               G_.accessor<scalar_t, 1>(),
               F.accessor<scalar_t, 1>(),
               I.accessor<int, 1>(),
               numel);
         }));
   }
}



// =================================
// ========== Quick Update =========
// =================================
template <typename scalar_t>
__global__ void alice_quick_update_kernel(
   torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> P,
   torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> M,
   const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> G,
   const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> F,
   const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> I,
   const int numel) {
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numel) {
      scalar_t g0 = G[i];
      const scalar_t v = g0 * g0;
      // Add regularization terms after second moment.
      g0 += (M[i][MNu] < 0 ? -F[FW1] : F[FW1]) + F[FW2] * M[i][MNu];
      // This is the standard update of the running gradient and second moment used in Adam.
      M[i][MV] = v   + F[FBet2] * (M[i][MV] - v);
      M[i][MG] = g0  + F[FBet1] * (M[i][MG] - g0);
   }
}

template <typename scalar_t>
void alice_quick_update_cpu(
   torch::TensorAccessor<scalar_t, 1> P,
   torch::TensorAccessor<scalar_t, 2> M,
   const torch::TensorAccessor<scalar_t, 1> G,
   const torch::TensorAccessor<scalar_t, 1> F,
   const torch::TensorAccessor<int, 1> I,
   const int numel) {
   for (int i = 0; i < numel; i++) {
      scalar_t g0 = G[i];
      const scalar_t v = g0 * g0;
      // Add regularization terms after second moment.
      g0 += (M[i][MNu] < 0 ? -F[FW1] : F[FW1]) + F[FW2] * M[i][MNu];
      M[i][MV] = v   + F[FBet2] * (M[i][MV] - v);
      M[i][MG] = g0  + F[FBet1] * (M[i][MG] - g0);
   }
}

void quick_update(
   torch::Tensor P,
   torch::Tensor M,
   const torch::Tensor G,
   const torch::Tensor F,
   const torch::Tensor I) {
   const int numel = M.size(0);
   assert(G.numel() == numel);
   assert(F.numel() == FLen);
   assert(I.numel() == ILen);

   torch::Tensor P_ = P.flatten();
   const torch::Tensor G_ = G.flatten();

   if (P_.device().is_cuda()) {
      CHECK_CONTIGUOUS(P_);
      CHECK_CUDA_INPUT(M);
      CHECK_CUDA_INPUT(G_);
      CHECK_CUDA_INPUT(F);
      CHECK_CUDA_INPUT(I);

      const int threads = THREADS_PER_SM;
      const dim3 blocks((numel + threads - 1) / threads);
      
      AT_DISPATCH_FLOATING_TYPES(
         M.scalar_type(), "quick_update_cuda",
         ([&] {
            alice_quick_update_kernel<scalar_t><<<blocks, threads>>>(
               P_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               M.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
               G_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               F.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               I.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
               numel);
         }));
   } else {
      CHECK_CPU_INPUT(P_);
      CHECK_CPU_INPUT(M);
      CHECK_CPU_INPUT(G_);
      CHECK_CPU_INPUT(F);
      CHECK_CPU_INPUT(I);

      AT_DISPATCH_FLOATING_TYPES(
         M.scalar_type(), "quick_update_cpu",
         ([&] {
            alice_quick_update_cpu<scalar_t>(
               P_.accessor<scalar_t, 1>(),
               M.accessor<scalar_t, 2>(),
               G_.accessor<scalar_t, 1>(),
               F.accessor<scalar_t, 1>(),
               I.accessor<int, 1>(),
               numel);
         }));
   }
}



// =================================
// ========== Update State =========
// =================================
template <typename scalar_t>
__global__ void alice_state_update_kernel(
   torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> P,
   torch::PackedTensorAccessor32<scalar_t,2,torch::RestrictPtrTraits> M,
   const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> F,
   const torch::PackedTensorAccessor32<int,1,torch::RestrictPtrTraits> I,
   const int numel) {
   const int i = blockIdx.x * blockDim.x + threadIdx.x;
   if (i < numel) {
      // At the moment, this retains no useful information and is unused until the next step.
      M[i][MZ] = 0;
      // L = L(mu) + delta^T * (g + 0.5 h * delta + 0.46 * sqrt(|delta| * rho))
      const scalar_t g = M[i][MG];
      const scalar_t absg = abs(g);
      const scalar_t rho = (I[IR] ? M[i][MR] : 0);
      const scalar_t h = (I[IH] == 0 ? 0 : (I[IH] == 1 ? M[i][MH] : sqrt(M[i][MH])));
      // Alg 3, Step 1.
      const scalar_t alpha = 0.238732 * rho / (absg + F[FEps]);
      // Alg 3, Step 2.
      const scalar_t h_opt = alpha + h + sqrt(alpha * (alpha + 2.0 * h)) + F[FEps];
      // Alg 3, Step 3.
      scalar_t delta = absg / h_opt;

      // Alg 3, Steps 4-6
      scalar_t delta_min = F[FTau1];
      scalar_t delta_max = F[FTau2];
      // I[IL] == 0, fixed scale. Nothing to do.
      if (I[IL] == 1) {
         // SGDM-based step bounds:
         delta_min *= absg;
         delta_max *= absg;
      } else if (I[IL] == 2) {
         // Adam-based step bounds:
         const scalar_t adam_factor = absg / (sqrt(M[i][MV]) + F[FEps]);
         delta_min *= adam_factor;
         delta_max *= adam_factor;
      }
      // Alg 3, Step 7.
      if (delta < delta_min) {
         delta = delta_min;
      } else if (!(delta < delta_max)) {
         // This comparison construction ensures that if delta is NaN, it is replaced by delta_max.
         delta = delta_max;
      }
      // Alg 3, Step 8.
      // Correct the sign for descent.
      delta = (g > 0 ? -delta : delta);
      // Look-ahead evalaution is updated from current position mu:
      const scalar_t mu = M[i][MMu];
      // Alg 3, Steps 9 and 10.
      M[i][MNu] = mu + F[FOmg] * delta;
      M[i][MMu] = mu + F[FPhi] * delta;
   }
}

template <typename scalar_t>
void alice_state_update_cpu(
   torch::TensorAccessor<scalar_t, 1> P,
   torch::TensorAccessor<scalar_t, 2> M,
   const torch::TensorAccessor<scalar_t, 1> F,
   const torch::TensorAccessor<int, 1> I,
   const int numel) {
   for (int i = 0; i < numel; i++) {
      // At the moment, this retains no useful information and is unused until the next step.
      M[i][MZ] = 0;
      // L = L(mu) + delta^T * (g + 0.5 h * delta + 0.46 * sqrt(|delta| * rho))
      const scalar_t g = M[i][MG];
      const scalar_t absg = abs(g);
      const scalar_t rho = (I[IR] ? M[i][MR] : 0);
      const scalar_t h = (I[IH] == 0 ? 0 : (I[IH] == 1 ? M[i][MH] : sqrt(M[i][MH])));
      const scalar_t alpha = 0.238732 * rho / (absg + F[FEps]);
      const scalar_t h_opt = alpha + h + sqrt(alpha * (alpha + 2.0 * h)) + F[FEps];
      scalar_t delta = absg / h_opt;

      scalar_t delta_min = F[FTau1];
      scalar_t delta_max = F[FTau2];
      // I[IL] == 0, fixed scale. Nothing to do.
      if (I[IL] == 1) {
         // SGDM-based step bounds:
         delta_min *= absg;
         delta_max *= absg;
      } else if (I[IL] == 2) {
         // Adam-based step bounds:
         const scalar_t adam_factor = absg / (sqrt(M[i][MV]) + F[FEps]);
         delta_min *= adam_factor;
         delta_max *= adam_factor;
      }
      if (delta < delta_min) {
         delta = delta_min;
      } else if (!(delta < delta_max)) {
         // This comparison construction ensures that if delta is NaN, it is replaced by delta_max.
         delta = delta_max;
      }
      // Correct the sign for descent.
      delta = (g > 0 ? -delta : delta);
      // Look-ahead evalaution is updated from current position mu:
      const scalar_t mu = M[i][MMu];
      M[i][MNu] = mu + F[FOmg] * delta;
      M[i][MMu] = mu + F[FPhi] * delta;
   }
}

void state_update(
   torch::Tensor P,
   torch::Tensor M,
   const torch::Tensor F,
   const torch::Tensor I) {
   const int numel = M.size(0);
   assert(I.numel() == ILen);
   torch::Tensor P_ = P.flatten();

   if (P_.device().is_cuda()) {
      CHECK_CUDA_INPUT(M);
      CHECK_CUDA_INPUT(F);
      CHECK_CUDA_INPUT(I);

      const int threads = THREADS_PER_SM;
      const dim3 blocks((numel + threads - 1) / threads);

      AT_DISPATCH_FLOATING_TYPES(
         M.scalar_type(), "state_update_cuda",
         ([&] {
            alice_state_update_kernel<scalar_t><<<blocks, threads>>>(
               P_.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               M.packed_accessor32<scalar_t,2,torch::RestrictPtrTraits>(),
               F.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
               I.packed_accessor32<int,1,torch::RestrictPtrTraits>(),
               numel);
         }));
   } else {
      CHECK_CPU_INPUT(P);
      CHECK_CPU_INPUT(M);
      CHECK_CPU_INPUT(F);
      CHECK_CPU_INPUT(I);

      AT_DISPATCH_FLOATING_TYPES(
         M.scalar_type(), "state_update_cpu",
         ([&] {
            alice_state_update_cpu<scalar_t>(
               P_.accessor<scalar_t, 1>(),
               M.accessor<scalar_t, 2>(),
               F.accessor<scalar_t, 1>(),
               I.accessor<int, 1>(),
               numel);
         }));
   }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_pert", &init_pert, "Set parameter perturbation.");
  m.def("pert_update", &pert_update, "Update state information from full perturbations.");
  m.def("quick_update", &quick_update, "Update gradient and second moment only.");
  m.def("state_update", &state_update, "Update optimization state.");
}

