
library(tidyverse)

data <- read.table("driver22.csv", header = T, sep = "") %>%
    mutate(speed = 22) %>%
    rbind(read.table("driver33.csv", header = T, sep = "") %>%
          mutate(speed = 33))

## Plot reward as the function of modeltime, between speed conditions.
ggplot(data %>%
       mutate(modeltime = round(modeltime, 0)) %>%
       group_by(speed, modeltime) %>%
       summarise(reward = mean(reward)) %>%
       ungroup() %>%
       mutate(modeltime = ntile(modeltime, 100)) %>%
       group_by(speed, modeltime) %>%
       summarise(reward = mean(reward)),
       aes(modeltime, reward, colour = factor(speed))) +
    geom_point() +
    geom_line()

data <- read.table("driver22_o9.csv", header = T, sep = "") %>%
    mutate(speed = 22, obs = 0.9) %>%
    rbind(read.table("driver33_o9.csv", header = T, sep = "") %>%
          mutate(speed = 33, obs = 0.9)) %>%
    rbind(read.table("driver22_o3.csv", header = T, sep = "") %>%
          mutate(speed = 22, obs = 0.3)) %>%
    rbind(read.table("driver33_o3.csv", header = T, sep = "") %>%
          mutate(speed = 33, obs = 0.3)) 

## Plot average lane deviation by speed x obs_prob.
ggplot(data %>%
       # aggregate into bins
       mutate(modeltime = ntile(modeltime, 10), pos = abs(pos)) %>%
       group_by(speed, modeltime, obs) %>%
       summarise(pos = mean(pos)),
       aes(modeltime, pos, colour = factor(speed))) +
    geom_point() +    
    geom_line() +
    facet_grid(. ~ obs)
