
library(data.table)
library(fixest)
library(hdm)
library(grf)

df <- fread("D:\\Downloads\\570hw\\R-code\\cleaned-usa.csv")

df_clean <- na.omit(df)
print(paste("useful data:", nrow(df_clean)))

# Baseline OLS Pooled

model_baseline <- feols(VEHICLES ~ WFH + HHINCOME + RACE_White + WHITE_COLLAR | PUMA + YEAR, data = df_clean)

cat("\n=== Baseline OLS  ===\n")
print(summary(model_baseline))

# 2019

df_2019 <- df_clean[YEAR == 2019]
model_placebo <- feols(VEHICLES ~ WFH + HHINCOME + RACE_White + WHITE_COLLAR | PUMA, data = df_2019)

cat("\n=== Just 2019 ===\n")
print(summary(model_placebo))





# ----------------DID---------------

# 2021
df_2021 <- df_clean[YEAR == 2021]
model_2021 <- feols(VEHICLES ~ WFH + HHINCOME + RACE_White + WHITE_COLLAR | PUMA, data = df_2021)

cat("\n=== 2021 ===\n")
print(summary(model_2021))


# DiD

df_clean$POST_COVID <- ifelse(df_clean$YEAR == 2021, 1, 0)

model_did <- feols(VEHICLES ~ WFH + POST_COVID + WFH:POST_COVID + HHINCOME + RACE_White + WHITE_COLLAR | PUMA, data = df_clean)

print(summary(model_did))



# ==========================plot(DiD)====================
library(ggplot2)
library(dplyr)

did_plot_data <- df_clean %>%
  group_by(YEAR, WFH) %>%
  summarise(
    mean_veh = mean(VEHICLES, na.rm = TRUE),
    se_veh = sd(VEHICLES, na.rm = TRUE) / sqrt(n()),
    .groups = 'drop'
  ) %>%
  mutate(
    WFH_Label = ifelse(WFH == 1, "WFH Group", "Non-WFH Group"),
    YEAR = as.factor(YEAR)
  )

fig1_did <- ggplot(did_plot_data, aes(x = YEAR, y = mean_veh, group = WFH_Label, color = WFH_Label, shape = WFH_Label)) +
  geom_line(size = 1.2) +
  geom_point(size = 4) +
  geom_errorbar(aes(ymin = mean_veh - 1.96*se_veh, ymax = mean_veh + 1.96*se_veh), width = 0.05, size = 0.8) +
  scale_color_manual(values = c("Non-WFH Group" = "blue", "WFH Group" = "red")) +
  labs(
    title = "Parallel Trends and DiD Marginal Effects",
    x = "Year",
    y = "Average Number of Vehicles",
    color = "Group",
    shape = "Group"
  ) +
  theme_minimal() +
  theme(
    legend.position = "bottom",
    text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

print(fig1_did)





# Double Lasso

Y <- df_clean$VEHICLES
D <- df_clean$WFH

X_matrix <- model.matrix(~ . - VEHICLES - WFH - SERIAL - CBSERIAL - YEAR - PUMA - HHWT - LOG_HHWT - 1, data = df_clean)

cat("\nRunning Double Lasso, large samples may take a few minutes...\n")
lasso_model <- rlassoEffect(x = X_matrix, y = Y, d = D, method = "double selection")

cat("\n=== Table 4: Double Lasso Results ===\n")
print(summary(lasso_model))


# ------------------lasso plot

library(ggplot2)

comparison_data <- data.frame(
  Method = c("OLS Baseline", "DiD Model", "Double Lasso (DML)"),
  Estimate = c(-0.438, -0.428, -1.556),
  Lower = c(-0.446, -0.434, -1.598), 
  Upper = c(-0.430, -0.422, -1.514)
)

fig3_compare <- ggplot(comparison_data, aes(x = Method, y = Estimate)) +
  geom_point(size = 4, color = "blue") +
  geom_errorbar(aes(ymin = Lower, ymax = Upper), width = 0.2, color = "blue") +
  geom_hline(yintercept = 0, linetype = "dashed", color = "red") +
  labs(title = "Robustness of WFH Effect Across Models",
       y = "Coefficient of WFH", x = "Estimation Method") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5, face = "bold"))

print(fig3_compare)




# Causal Forest

set.seed(999) 
sample_idx <- sample(1:nrow(df_clean), 100000)

Y_cf <- Y[sample_idx]
W_cf <- D[sample_idx] 
X_cf <- X_matrix[sample_idx, ]

# casual forest----------

cols_to_keep <- !grepl("WFH", colnames(X_cf))
X_cf_clean <- X_cf[, cols_to_keep]

cat("No WFH, clean data:", ncol(X_cf_clean), "\n")

cat("\n planting trees...\n")
cf_model_clean <- causal_forest(X = X_cf_clean, Y = Y_cf, W = W_cf)

ate_clean <- average_treatment_effect(cf_model_clean)
cat("\n=== Causal Forest: ATE ===\n")
print(ate_clean)

var_imp_clean <- variable_importance(cf_model_clean)
var_importance_df_clean <- data.frame(Feature = colnames(X_cf_clean), Importance = as.vector(var_imp_clean))
var_importance_df_clean <- var_importance_df_clean[order(-var_importance_df_clean$Importance), ]

cat("\n=== Importance Variables ===\n")
print(head(var_importance_df_clean, 5))


# ----------------------plot for Importance Variables---------------


library(ggplot2)
library(dplyr)

plot_data <- head(var_importance_df_clean, 5) %>%
  arrange(Importance) %>%
  mutate(Feature = factor(Feature, levels = Feature))

plot_data$Feature <- recode(plot_data$Feature,
                            "HHINCOME" = "Household Income",
                            "HHINCOME_LOG" = "Log(Household Income)",
                            "PUMA_WHITE_RATIO" = "PUMA White Ratio",
                            "PUMA_WHITE_COLLAR_JOB_RATIO" = "PUMA White-collar Ratio",
                            "RACE_CATEGORY.3White" = "Race: White")

fig4_var_imp <- ggplot(plot_data, aes(x = Feature, y = Importance)) +
  geom_bar(stat = "identity", fill = "lightblue", width = 0.6) +
  coord_flip() + 
  geom_text(aes(label = sprintf("%.3f", Importance)), hjust = -0.2, size = 4) +
  labs(
    title = "Variable Importance in Causal Forest",
    x = "Features",
    y = "Importance Score"
  ) +
  theme_minimal() +
  ylim(0, max(plot_data$Importance) * 1.1) + 
  theme(
    text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    axis.text.y = element_text(face = "bold")
  )

print(fig4_var_imp)


# ----------------plot: heterogeneous--------------

library(ggplot2)
library(dplyr)

heterogeneity_results_en <- heterogeneity_results %>%
  mutate(
    Income_Group_En = ifelse(Income_Group == "High Income (>= Median)", 
                             "High Income (>= Median)", 
                             "Low Income (< Median)")
  )

fig2_cate <- ggplot(heterogeneity_results_en, aes(x = Income_Group_En, y = Average_CATE, fill = Income_Group_En)) +
  geom_bar(stat = "identity", width = 0.5, color = "black", alpha = 0.8) +
  geom_text(aes(label = sprintf("%.3f", Average_CATE)), 
            vjust = ifelse(heterogeneity_results_en$Average_CATE < 0, 1.5, -0.5), 
            size = 5) +
  scale_fill_manual(values = c("High Income (>= Median)" = "grey", 
                               "Low Income (< Median)" = "orange")) +
  labs(
    title = "Heterogeneous Treatment Effects by Income",
    x = "Income Group",
    y = "Conditional Average Treatment Effect (CATE)"
  ) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black", size = 1) +
  theme_minimal() +
  theme(
    legend.position = "none", 
    text = element_text(size = 12),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

print(fig2_cate)


# heterogeneous

cate_predictions <- predict(cf_model_clean)$predictions
df_sample_cf <- df_clean[sample_idx, ] 
df_sample_cf$CATE <- cate_predictions

median_income <- median(df_sample_cf$HHINCOME, na.rm = TRUE)

df_sample_cf$Income_Group <- ifelse(df_sample_cf$HHINCOME >= median_income, 
                                    "High Income (>= Median)", 
                                    "Low Income (< Median)")

library(dplyr)
heterogeneity_results <- df_sample_cf %>%
  group_by(Income_Group) %>%
  summarise(
    Average_CATE = mean(CATE, na.rm = TRUE),
    Average_HHINCOME = mean(HHINCOME, na.rm = TRUE),
    Sample_Size = n()
  )

cat("\n=== heterogeneous: rich vs poor of WFH effect ===\n")
print(heterogeneity_results)




# =====================robustness==========================
prop_scores <- cf_model_clean$W.hat

strict_indices <- which(prop_scores >= 0.1 & prop_scores <= 0.9)
X_strict <- X_cf[strict_indices, ]
Y_strict <- Y_cf[strict_indices]
W_strict <- W_cf[strict_indices]

cat("original samples:", length(Y_cf), "\n")
cat("filtered samples:", length(Y_strict), "\n")

library(grf)
cf_model_strict <- causal_forest(X_strict, Y_strict, W_strict, seed = 999)


ate_overlap <- average_treatment_effect(cf_model_clean, target.sample = "overlap")

cat("\n=== Overlap Weights (ATO) ===\n")
print(ate_overlap)

