-- crypto_trading.training_summaries definition

CREATE TABLE `training_summaries` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `crypto_code` varchar(20) NOT NULL,
  `training_loss` float NOT NULL,
  `epochs` int(11) NOT NULL,
  `batch_size` int(11) NOT NULL,
  `n_steps` int(11) NOT NULL,
  `lstm_units` int(11) NOT NULL,
  `dropout` float NOT NULL,
  `num_samples` int(11) NOT NULL,
  `model_path` varchar(255) NOT NULL,
  `trained_at` datetime NOT NULL,
  `created_at` datetime NOT NULL DEFAULT utc_timestamp(),
  PRIMARY KEY (`created_at`,`crypto_code`),
  UNIQUE KEY `id` (`id`),
  KEY `crypto_code` (`crypto_code`),
  CONSTRAINT `crypto_trading_data_ibfk_1` FOREIGN KEY (`crypto_code`) REFERENCES `crypto_currencies` (`code`)
) ENGINE=InnoDB AUTO_INCREMENT=41545 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;