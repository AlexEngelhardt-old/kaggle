library(tidyverse)
library(mlr)
library(xgboost)

config <- list(
    full_train_data = FALSE,
    full_test_data = TRUE
)

n_max_train <- ifelse(config$full_train_data, Inf, 100000)
n_max_test <- ifelse(config$full_test_data, Inf, 100000)

train <- read_csv("data/train.csv", col_types = "cccccccl", n_max = n_max_train)
test <- read_csv("data/test.csv", col_types = "icccccc", n_max = n_max_test)

test$app <- as.factor(test$app)

means <- train %>%
    group_by(app) %>%
    summarize(yhat = mean(is_attributed))

# avoid zeros and ones
means$yhat[means$yhat == 0] <- min(means$yhat[means$yhat > 0])
means$yhat[means$yhat == 1] <- max(means$yhat[means$yhat < 1])

test$is_attributed <- means$yhat[match(test$app, means$app)]

test$is_attributed[is.na(test$is_attributed)] <- median(test$is_attributed, na.rm=TRUE)

# reduce file size (500MB) of submission by rounding floats:
test$is_attributed <- round(test$is_attributed, 5)

test %>%
    select(click_id, is_attributed) %>%
    write_csv("data/submissions/00_mvp.csv")
