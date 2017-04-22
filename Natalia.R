
library(tidyverse)

# Create sample with real .10 difference
treatment = rbinom(1000, 1, .53)
control = rbinom(1000, 1, .5)

mean(treatment) - mean(control)

# T-test (Not supposed to use because this is binomial data but works pretty much fine)
t.test(treatment, control)

# Chi-square 2 proportion test (use for binomial data)
prop.test(c(sum(treatment), sum(control)), c(1000, 1000), correct = TRUE)

# Permutation test (makes no assumptions about type of data or distributions)
num_iterations = 100000
empirical_distribution = NULL

for (i in 1:num_iterations) {
  all_data = sample(c(treatment, control))
  empirical_distribution = append(empirical_distribution, mean(all_data[1:1000]) - mean(all_data[1001:2000]))
}

# Plot it
hist(empirical_distribution, col = "black", breaks = 100, xlim = c(-(mean(treatment - control)) * 1.5, mean(treatment - control) * 1.5))
abline(v = mean(treatment - control), col = "blue", lwd = 2)

# Get p value
sum(abs(empirical_distribution) > abs(mean(treatment - control))) / num_iterations



