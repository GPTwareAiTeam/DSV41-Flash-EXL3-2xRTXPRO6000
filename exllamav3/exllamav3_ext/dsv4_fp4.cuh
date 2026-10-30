#pragma once
#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> dsv4_fp4_quant(const at::Tensor& input, bool indexer);
at::Tensor dsv4_fp4_dequant(const at::Tensor& packed, const at::Tensor& scales, bool indexer);
