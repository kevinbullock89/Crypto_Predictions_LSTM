-- crypto_trading.predicted_crypto_prices definition

CREATE TABLE `predicted_crypto_prices` (
  `id` bigint(20) NOT NULL AUTO_INCREMENT,
  `datetime` datetime NOT NULL,
  `currency_code` varchar(10) NOT NULL,
  `actual_price` decimal(18,8) DEFAULT NULL,
  `predicted_price` decimal(18,8) DEFAULT NULL,
  `created_at` timestamp NULL DEFAULT current_timestamp(),
  PRIMARY KEY (`id`),
  UNIQUE KEY `unique_idx_currency_time` (`currency_code`,`datetime`,`created_at`),
  CONSTRAINT `crypto_data_predict_1` FOREIGN KEY (`currency_code`) REFERENCES `crypto_currencies` (`code`)
) ENGINE=InnoDB AUTO_INCREMENT=473807 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;