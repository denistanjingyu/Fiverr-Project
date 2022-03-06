### Task: Visualise and describe the data via unsupervised learning methods. ###
################################################################################

# Load the data and remove the rows with missing data (NA)
ewcs = read.table("C:\\Users\\user\\Desktop\\Project\\Part 1\\Dataset\\EWCS_2016.csv", sep = ",", header = TRUE)
ewcs[,][ewcs[,,] == -999] <- NA
kk = complete.cases(ewcs)
ewcs = ewcs[kk,]

# Check the internal data structure (All int data types)
# Examine the basic descriptive statistics of each column (e.g. Min, 1st Qu., Median, Mean...)
# Observation: All survey questions have median of 2 (Answer: Most of the time) except the last question (Q90f)
# with median of 1 (Answer: Always)
str(ewcs)
summary(ewcs)

# Principal component analysis (PCA) on the EWCS data
# Use column 3 to 11 which represent the survey data
# Center and scale the data 
# PCA is sensitive to the relative scaling of the original variables
# If one component varies less than another because of their respective scales (1000s vs. 10s range), 
# PCA might determine that the direction of maximal variance more closely corresponds with the larger range (1000s) axis, 
# if those features are not scaled.
ewcs.pca <- prcomp(ewcs[c(3:11)], center = TRUE, scale = TRUE)

# Standard deviation: eigenvalue after centering and scaling
# Proportion of Variance: amount of variance the component accounts for in the data (PC1 accounts for 48.67% of total variance in the data)
# Cumulative Proportion: accumulated amount of explained variance (first 5 components account for >80% of total variance in the data)
summary(ewcs.pca)

# Calculate total variance explained by each principal component
var_explained = ewcs.pca$sdev^2 / sum(ewcs.pca$sdev^2)

# Create scree plot using ggplot2
library(ggplot2)
# Shows variance explained by each principal component
scree_plot <- qplot(c(1:9), var_explained) + 
  geom_line() + 
  xlab("Principal Component") + 
  ylab("Variance Explained") +
  ggtitle("Scree Plot") +
  ylim(0, 1)
scree_plot + scale_x_continuous(breaks = seq(1, 9, by = 1))

# Calculate cumulative variance
cumpro <- cumsum(ewcs.pca$sdev^2 / sum(ewcs.pca$sdev^2))

# Create cumulative variance plot
# We can use just 5 principal components to explain >80% of variance
cum_var_plot <- plot(cumpro[1:9], xlab = "Principal Component", ylab = "Amount of explained variance", main = "Cumulative variance plot") 
axis(1, at = seq(1, 9, by = 1))
abline(v = 5, col = "blue", lty = 5)
abline(h = 0.84605, col = "blue", lty = 5)
legend("topleft", legend = c("Cut-off at Principal Component 5"),
       col = c("blue"), lty = 5, cex = 0.8)

# Further reduce principal components to 2 for visualization purposes (2d)
# 2 PCs can explain >60% of total variance in the data
# Use gender to distinguish the data points in the visualization
# factoextra: Extract and Visualize the Results of Multivariate Data Analyses
library("FactoMineR")
library("factoextra")

# Change gender (Q2a) to factor type (categorical)
ewcs$Q2a <- as.factor(ewcs$Q2a)

# No visible difference between male and female in survey answers (ellipses pretty much overlap each other)
fviz_pca_ind(ewcs.pca, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = ewcs$Q2a, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Gender") +
  ggtitle("2D PCA-plot from 9 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))


# Categorize age (Q2b) into Working-age (15-64yo) and Elderly (>64yo)
ewcs["AgeGroup"] <- cut(ewcs$Q2b, breaks = c(15, 64, 87), labels = c("Working-age", "Elderly"), include.lowest = TRUE)

# Slight separation of working-age and elderly group but not significant
fviz_pca_ind(ewcs.pca, geom.ind = "point", pointshape = 21, 
             pointsize = 2, 
             fill.ind = ewcs$AgeGroup, 
             col.ind = "black", 
             palette = "jco", 
             addEllipses = TRUE,
             label = "var",
             col.var = "black",
             repel = TRUE,
             legend.title = "Age Group") +
  ggtitle("2D PCA-plot from 9 feature dataset") +
  theme(plot.title = element_text(hjust = 0.5))

# PCA matrix to obtain the loadings
pca.mat = function(loading, comp.sdev){
  loading*comp.sdev
}

# Define the loading and standard deviation
loading = ewcs.pca$rotation
comp.sdev = ewcs.pca$sdev

# Principal component variables (apply the function with the loading and std dev)
pc.var = t(apply(loading, 1, pca.mat, comp.sdev)) 
# Display the PCA matrix (PC1 & PC2)
pc.var[,1:2] 

library(ggfortify)

# PLot PCA biplot
# Q87 series have higher loadings in PC1 while Q90 series have higher loadings in PC2
# Q87 vectors are in the same direction as the dispersion of Q90f (1/2). 
# If an individual were to respond positively on Q90f series, Q87 would be answered positively as well.
# Always (1) and Most of the time (2) are observed to be denser relative to the negative responses (4 and 5) having wider spread.
autoplot(ewcs.pca, colour = 'Q90f', loadings = TRUE, 
         loadings.label = TRUE, loadings.label.size = 5)

# First 2 principal components and the loadings
# Parameter scale = 0 ensures that arrows are scaled to represent the loadings
biplot(ewcs.pca, scale = 0, col = c("white", "deeppink3"))
