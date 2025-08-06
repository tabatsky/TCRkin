library(ggplot2)
library(png)
library(reshape2)

peptides = c("N4", "T4", "V4", "UVpep")

in_file1 <- paste("results_fit10_", "peakL1", ".csv", sep="")
in_file2 <- paste("results_fit10_", "peakL2", ".csv", sep="")
in_file3 <- paste("results_fit10_", "peakL3", ".csv", sep="")
out_file <- paste("boxplot_", "peak3", ".png", sep="")

data1 <- read.csv(in_file1,
                 header = TRUE, sep = ";")
data_frame1 = data.frame(data1)
data2 <- read.csv(in_file2,
                 header = TRUE, sep = ";")
data_frame2 = data.frame(data2)
data3 <- read.csv(in_file3,
                 header = TRUE, sep = ";")
data_frame3 = data.frame(data3)
data_frame = data_frame1

vars <- list()

for (peptide in peptides) {
  v = paste(peptide, '_L1', sep="")
  vars <- append(vars, v)
  data_frame[v] = data_frame1[peptide]
  v = paste(peptide, '_L2', sep="")
  vars <- append(vars, v)
  data_frame[v] = data_frame2[peptide]
  v = paste(peptide, '_L3', sep="")
  vars <- append(vars, v)
  data_frame[v] = data_frame3[peptide]
}
data_mod <- melt(data_frame, id.vars='fileName',  
                 measure.vars=vars)
data_mod$value = sapply(data_mod$value, function(x) log10(as.numeric(x)))

colors <- list("#ff0000", "#ff0000", "#ff0000", "#ff00ff", "#ff00ff", "#ff00ff", "#00ff00", "#00ff00", "#00ff00", "#0000ff", "#0000ff", "#0000ff")

box_plot <- ggplot(data_mod) +
  geom_boxplot(aes(x=variable, y=value), fill=colors) +
  coord_cartesian(ylim = c(-4, 1)) +
  labs(x = '', y = paste('log10(', 'peak', ')', sep=''), color='') +
  theme_bw() +
  theme(plot.title = element_text(size = 18), 
        text = element_text(size=18),
        axis.text = element_text(size=18),
        axis.text.x = element_text(size=18, angle=90, hjust=1.0),
        axis.title.x = element_text(size=18),
        axis.title.y = element_text(size=18),
        panel.grid.major = element_blank(), 
        panel.grid.minor = element_blank()
  ) 
png(out_file)
print(box_plot)
dev.off()

