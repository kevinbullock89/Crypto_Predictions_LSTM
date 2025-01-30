-- crypto_trading.loading_information definition

CREATE TABLE `loading_information` (
  `Currency_Code` varchar(10) NOT NULL,
  `Last_Loading_Datetime` datetime NOT NULL,
  PRIMARY KEY (`Currency_Code`),
  CONSTRAINT `loading_information_ibfk_1` FOREIGN KEY (`Currency_Code`) REFERENCES `crypto_currencies` (`code`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;