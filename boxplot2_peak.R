library(ggplot2)
library(png)
library(reshape2)

peptides = c("N4", "T4", "V4", "UVpep")

in_file <- paste("results_fit10_", "peak", ".csv", sep="")
out_file <- paste("boxplot_", "peak", ".png", sep="")

data <- read.csv(in_file,
                 header = TRUE, sep = ";")
data_frame = data.frame(data)

vars <- list()

for (peptide in peptides) {
  for (ing in c("", "_antiCD8")) {
    v = paste(peptide, ing, sep="")
    vars <- append(vars, v)
  }
}
data_mod <- melt(data_frame, id.vars='fileName',  
                 measure.vars=vars)
data_mod$value = sapply(data_mod$value, function(x) log10(as.numeric(x)))

colors <- list("#ff0000", "#ffbbbb", "#ffff00", "#ffffcc", "#00ff00", "#bbffbb", "#0000ff", "#bbbbff")

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

