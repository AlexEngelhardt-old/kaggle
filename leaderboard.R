library(tidyverse)
library(readr)
submissions = read_csv("jigsaw-toxic-comment-classification-challenge-publicleaderboard.csv")

head(submissions)

lb <- submissions %>%
    group_by(TeamName) %>%
    summarize(score = max(Score)) %>%
    arrange(desc(score))

write_csv(lb, "leaderboard.csv")
