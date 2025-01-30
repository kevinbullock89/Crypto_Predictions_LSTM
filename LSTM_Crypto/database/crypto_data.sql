-- crypto_trading.crypto_data definition

CREATE TABLE `crypto_data` (
  `EntryID` bigint(20) NOT NULL AUTO_INCREMENT,
  `Datetime` datetime NOT NULL,
  `Open` decimal(18,2) DEFAULT NULL,
  `High` decimal(18,2) DEFAULT NULL,
  `Low` decimal(18,2) DEFAULT NULL,
  `Close` decimal(18,2) DEFAULT NULL,
  `Volume` bigint(20) DEFAULT NULL,
  `Dividends` decimal(18,2) DEFAULT NULL,
  `StockSplits` int(11) DEFAULT NULL,
  `Currency_Code` varchar(10) NOT NULL,
  PRIMARY KEY (`Datetime`,`Currency_Code`),
  UNIQUE KEY `EntryID` (`EntryID`),
  KEY `Currency_Code` (`Currency_Code`),
  CONSTRAINT `crypto_data_ibfk_1` FOREIGN KEY (`Currency_Code`) REFERENCES `crypto_currencies` (`code`)
) ENGINE=InnoDB AUTO_INCREMENT=79393 DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;