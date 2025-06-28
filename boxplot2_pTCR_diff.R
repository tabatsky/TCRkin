library(ggplot2)
library(png)
library(reshape2)

peptides = c("N4", "T4", "V4", "UVpep")

in_file <- "pTCR1.csv" # pTCR1.csv, pTCR2.csv, pTCR3.csv
out_file <- "boxplot_pTCR1_diff.png"

data <- read.csv(in_file,
                 header = TRUE, sep = ";")
data_frame = data.frame(data)

vars1 <- list()
vars2 <- list()

for (peptide in peptides) {
  vars1 <- append(vars1, peptide)
  vars2 <- append(vars2, paste0(peptide, "_antiCD8"))
}

data_mod1 <- melt(data_frame,  
                 measure.vars=vars1)
data_mod2 <- melt(data_frame,  
                  measure.vars=vars2)
data_mod <- data_mod1
data_mod$value <- data_mod1$value - data_mod2$value

colors <- list("#ff0000", "#ffff00", "#00ff00", "#0000ff")

box_plot <- ggplot(data_mod) +
  geom_boxplot(aes(x=variable, y=value), fill=colors) +
  coord_cartesian(ylim = c(0, 10000)) +
  labs(x = '', y = 'mean pTCR', color='') +
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