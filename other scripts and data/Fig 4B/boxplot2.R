library(ggplot2)
library(png)
library(reshape2)

peptides = c("N4", "T4", "V4", "UVpep")

kNames1 = c("k1", "k2", "k3")
kNames2 = c("kon", "koff", "krel")

for (i in c(1, 2, 3)) {
  kName = kNames1[i]
  kName2 = kNames2[i]
  in_file <- paste("results_fit10_", kName, ".csv", sep="")
  out_file <- paste("boxplot_", kName, ".png", sep="")
  
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
    coord_cartesian(ylim = c(-5, 1)) +
    labs(x = '', y = paste('log10(', kName2, ')', sep=''), color='') +
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
}

in_file_1 <- paste("results_fit10_k1.csv", sep="")
in_file_2 <- paste("results_fit10_k2.csv", sep="")
out_file <- paste("boxplot_kon_koff.png", sep="")

vars <- list()
for (peptide in peptides) {
  for (ing in c("", "_antiCD8")) {
    v = paste(peptide, ing, sep="")
    vars <- append(vars, v)
  }
}

data_1 <- read.csv(in_file_1,
                 header = TRUE, sep = ";")
data_frame_1 = data.frame(data_1)
data_mod_1 <- melt(data_frame_1, id.vars='fileName',  
                 measure.vars=vars)
data_mod_1$value = sapply(data_mod_1$value, function(x) log10(as.numeric(x)))

data_2 <- read.csv(in_file_2,
                   header = TRUE, sep = ";")
data_frame_2 = data.frame(data_2)
data_mod_2 <- melt(data_frame_2, id.vars='fileName',  
                   measure.vars=vars)
data_mod_2$value = sapply(data_mod_2$value, function(x) log10(as.numeric(x)))


data_mod <- melt(data_frame, id.vars='fileName',  
                 measure.vars=vars)
data_mod$value = data_mod_1$value - data_mod_2$value

#data_mod$value = sapply(data_mod$value, function(x) 10**x)

colors <- list("#ff0000", "#ffbbbb", "#ffff00", "#ffffcc", "#00ff00", "#bbffbb", "#0000ff", "#bbbbff")

box_plot <- ggplot(data_mod) +
  geom_boxplot(aes(x=variable, y=value), fill=colors) +
  coord_cartesian(ylim = c(-3, 0)) +
  labs(x = '', y = paste('log10(', 'kon/koff', ')', sep=''), color='') +
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
