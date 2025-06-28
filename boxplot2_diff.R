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
  out_file <- paste("boxplot_", kName, "_diff.png", sep="")
  
  data <- read.csv(in_file,
                   header = TRUE, sep = ";")
  data_frame = data.frame(data)
  
  vars1 <- list()
  vars2 <- list()
  
  for (peptide in peptides) {
    vars1 <- append(vars1, peptide)
    vars2 <- append(vars2, paste0(peptide, "_antiCD8"))
  }
  data_mod1 <- melt(data_frame, id.vars='fileName',  
                   measure.vars=vars1)
  data_mod2 <- melt(data_frame, id.vars='fileName',  
                    measure.vars=vars2)
  data_mod1$value = sapply(data_mod1$value, function(x) log10(as.numeric(x)))
  data_mod2$value = sapply(data_mod2$value, function(x) log10(as.numeric(x)))
  data_mod = data_mod1
  data_mod$value = data_mod1$value - data_mod2$value
  
  colors <- list("#ff0000", "#ffff00", "#00ff00", "#0000ff")
  
  box_plot <- ggplot(data_mod) +
    geom_boxplot(aes(x=variable, y=value), fill=colors) +
    coord_cartesian(ylim = c(-1, 3)) +
    labs(x = '', y = paste('log10(', kName2, ')', sep=''), color='') +
    theme_bw() +
    theme(plot.title = element_text(size = 24), 
          text = element_text(size=24),
          axis.text = element_text(size=24),
          axis.text.x = element_text(size=24, angle=90, hjust=1.0),
          axis.title.x = element_text(size=24),
          axis.title.y = element_text(size=24),
          panel.grid.major = element_blank(), 
          panel.grid.minor = element_blank()
    ) 
  png(out_file)
  print(box_plot)
  dev.off()
}

in_file_1 <- paste("results_fit10_k1.csv", sep="")
in_file_2 <- paste("results_fit10_k2.csv", sep="")
out_file <- paste("boxplot_kon_koff_diff.png", sep="")

vars1 <- list()
vars2 <- list()

for (peptide in peptides) {
  vars1 <- append(vars1, peptide)
  vars2 <- append(vars2, paste0(peptide, "_antiCD8"))
}

data_1 <- read.csv(in_file_1,
                 header = TRUE, sep = ";")
data_frame_1 = data.frame(data_1)
data_mod_11 <- melt(data_frame_1, id.vars='fileName',
                 measure.vars=vars1)
data_mod_12 <- melt(data_frame_1, id.vars='fileName',
                    measure.vars=vars2)
data_mod_11$value = sapply(data_mod_11$value, function(x) log10(as.numeric(x)))
data_mod_12$value = sapply(data_mod_12$value, function(x) log10(as.numeric(x)))
data_mod1 = data_mod_11
data_mod1$value = data_mod_11$value - data_mod_12$value

data_2 <- read.csv(in_file_2,
                   header = TRUE, sep = ";")
data_frame_2 = data.frame(data_2)
data_mod_21 <- melt(data_frame_2, id.vars='fileName',
                    measure.vars=vars1)
data_mod_22 <- melt(data_frame_2, id.vars='fileName',
                    measure.vars=vars2)
data_mod_21$value = sapply(data_mod_21$value, function(x) log10(as.numeric(x)))
data_mod_22$value = sapply(data_mod_22$value, function(x) log10(as.numeric(x)))
data_mod2 = data_mod_21
data_mod2$value = data_mod_21$value - data_mod_22$value


data_mod <- melt(data_frame, id.vars='fileName',
                 measure.vars=vars1)
data_mod$value = data_mod1$value - data_mod2$value

colors <- list("#ff0000", "#ffff00", "#00ff00", "#0000ff")

box_plot <- ggplot(data_mod) +
  geom_boxplot(aes(x=variable, y=value), fill=colors) +
  coord_cartesian(ylim = c(-3, 3)) +
  labs(x = '', y = paste('log10(', 'kon/koff', ')', sep=''), color='') +
  theme_bw() +
  theme(plot.title = element_text(size = 24),
        text = element_text(size=24),
        axis.text = element_text(size=24),
        axis.text.x = element_text(size=24, angle=90, hjust=1.0),
        axis.title.x = element_text(size=24),
        axis.title.y = element_text(size=24),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank()
  )
png(out_file)
print(box_plot)
dev.off()
