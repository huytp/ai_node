"""
Malicious URL Detection using Logistic Regression
Based on features extracted from URL characteristics
"""
import re
import urllib.parse
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import os


class URLFeatureExtractor:
    """Extract features from URL for malicious detection"""

    # Suspicious keywords commonly found in malicious URLs
    SUSPICIOUS_KEYWORDS = [
        'malware', 'virus', 'trojan', 'phishing', 'spam', 'scam',
        'fake', 'fraud', 'hack', 'exploit', 'crack', 'warez',
        'porn', 'adult', 'casino', 'gambling', 'bitcoin', 'crypto',
        'click', 'download', 'free', 'win', 'prize', 'lottery'
    ]

    # Suspicious TLDs
    SUSPICIOUS_TLDS = ['.tk', '.ml', '.ga', '.cf', '.gq', '.xyz', '.top']

    # URL shortening services
    SHORTENING_SERVICES = [
        'bit.ly', 'tinyurl.com', 'goo.gl', 't.co', 'ow.ly', 'is.gd',
        'buff.ly', 'adf.ly', 'short.link', 'tiny.cc'
    ]

    @staticmethod
    def extract_features(url: str) -> List[float]:
        """
        Extract features from URL

        Features:
        1. URL length
        2. Number of dots
        3. Number of hyphens
        4. Number of subdirectories
        5. Number of parameters
        6. Has IP address
        7. Has suspicious keywords
        8. Has suspicious TLD
        9. Is URL shortening service
        10. Number of special characters
        11. Ratio of digits to length
        12. Ratio of special chars to length
        """
        if not url:
            return [0.0] * 12

        url_lower = url.lower()
        parsed = urllib.parse.urlparse(url)
        domain = parsed.netloc or parsed.path.split('/')[0]
        path = parsed.path

        features = []

        # 1. URL length
        features.append(float(len(url)))

        # 2. Number of dots
        features.append(float(url.count('.')))

        # 3. Number of hyphens
        features.append(float(url.count('-')))

        # 4. Number of subdirectories (path depth)
        path_parts = [p for p in path.split('/') if p]
        features.append(float(len(path_parts)))

        # 5. Number of parameters
        query_params = urllib.parse.parse_qs(parsed.query)
        features.append(float(len(query_params)))

        # 6. Has IP address
        ip_pattern = r'\b(?:\d{1,3}\.){3}\d{1,3}\b'
        has_ip = bool(re.search(ip_pattern, domain))
        features.append(1.0 if has_ip else 0.0)

        # 7. Has suspicious keywords
        has_suspicious_keyword = any(
            keyword in url_lower
            for keyword in URLFeatureExtractor.SUSPICIOUS_KEYWORDS
        )
        features.append(1.0 if has_suspicious_keyword else 0.0)

        # 8. Has suspicious TLD
        has_suspicious_tld = any(
            domain.endswith(tld)
            for tld in URLFeatureExtractor.SUSPICIOUS_TLDS
        )
        features.append(1.0 if has_suspicious_tld else 0.0)

        # 9. Is URL shortening service
        is_shortening = any(
            service in domain
            for service in URLFeatureExtractor.SHORTENING_SERVICES
        )
        features.append(1.0 if is_shortening else 0.0)

        # 10. Number of special characters
        special_chars = len(re.findall(r'[^a-zA-Z0-9.\-/:]', url))
        features.append(float(special_chars))

        # 11. Ratio of digits to length
        digit_count = len(re.findall(r'\d', url))
        digit_ratio = digit_count / len(url) if url else 0.0
        features.append(digit_ratio)

        # 12. Ratio of special chars to length
        special_ratio = special_chars / len(url) if url else 0.0
        features.append(special_ratio)

        return features


class MaliciousURLDetector:
    """Malicious URL Detector using Logistic Regression"""

    def __init__(self, model_path: str = None, csv_path: str = None):
        self.feature_extractor = URLFeatureExtractor()
        self.model = None
        self.scaler = StandardScaler()
        self.model_path = model_path or os.path.join(
            os.path.dirname(__file__), 'url_detector_model.pkl'
        )
        self.csv_path = csv_path or os.path.join(
            os.path.dirname(__file__), 'data', 'urldata.csv'
        )
        self._initialize_model()

    def _initialize_model(self):
        """Initialize or load the model"""
        if os.path.exists(self.model_path):
            try:
                # Load existing model
                model_data = joblib.load(self.model_path)
                self.model = model_data['model']
                self.scaler = model_data['scaler']
                print(f"Loaded model from {self.model_path}")
                return
            except Exception as e:
                print(f"Error loading model: {e}. Creating new model.")

        # Create model
        self.model = LogisticRegression(random_state=42, max_iter=1000)

        # Try to train from CSV file, fallback to synthetic data
        if os.path.exists(self.csv_path):
            try:
                print(f"Found CSV file: {self.csv_path}")
                self._train_from_csv()
            except Exception as e:
                print(f"Error training from CSV: {e}. Using synthetic data.")
                self._train_synthetic_model()
        else:
            print(f"CSV file not found: {self.csv_path}. Using synthetic data.")
            self._train_synthetic_model()

    def _load_csv_data(self) -> Optional[tuple]:
        """
        Load data from CSV file

        Returns:
            (urls, labels) tuple or None if error
        """
        try:
            # Try to read CSV with different possible formats
            df = pd.read_csv(self.csv_path)

            # Print column names for debugging
            print(f"CSV columns: {df.columns.tolist()}")
            print(f"CSV shape: {df.shape}")

            # Try to find URL column (case-insensitive)
            url_col = None
            for col in df.columns:
                if col.lower() in ['url', 'link', 'website', 'domain']:
                    url_col = col
                    break

            if url_col is None:
                # Assume first column is URL
                url_col = df.columns[0]
                print(f"Using first column as URL: {url_col}")

            # Try to find label column (case-insensitive)
            label_col = None
            for col in df.columns:
                if col.lower() in ['label', 'type', 'class', 'category', 'result', 'status']:
                    label_col = col
                    break

            if label_col is None:
                # Assume second column is label
                if len(df.columns) > 1:
                    label_col = df.columns[1]
                    print(f"Using second column as label: {label_col}")
                else:
                    print("Warning: No label column found. Assuming all URLs are benign.")
                    return None

            # Extract URLs and labels
            urls_raw = df[url_col].astype(str).tolist()
            labels_raw = df[label_col].astype(str).tolist()

            # Normalize URLs and convert labels in one pass
            normalized_urls = []
            y = []

            for url, label in zip(urls_raw, labels_raw):
                # Skip if URL is invalid
                if pd.isna(url) or not str(url).strip():
                    continue

                url_str = str(url).strip()

                # Skip header row if present
                if url_str.lower() in ['url', 'link', 'website', 'domain']:
                    continue

                # Add protocol if missing
                if not url_str.startswith(('http://', 'https://')):
                    url_str = 'http://' + url_str

                # Convert label to binary
                label_lower = str(label).lower().strip()
                if label_lower in ['1', 'malicious', 'bad', 'phishing', 'malware', 'spam', 'true']:
                    y.append(1)
                elif label_lower in ['0', 'benign', 'good', 'legitimate', 'safe', 'false']:
                    y.append(0)
                else:
                    # Default: assume benign if unclear
                    y.append(0)

                normalized_urls.append(url_str)

            y = np.array(y)
            urls = normalized_urls

            print(f"Loaded {len(urls)} URLs from CSV")
            print(f"Malicious: {np.sum(y)}, Benign: {len(y) - np.sum(y)}")

            return urls, y

        except Exception as e:
            print(f"Error loading CSV: {e}")
            import traceback
            traceback.print_exc()
            return None

    def _train_from_csv(self):
        """Train model from CSV file"""
        data = self._load_csv_data()

        if data is None:
            raise ValueError("Failed to load CSV data")

        urls, y = data

        if len(urls) == 0:
            raise ValueError("No valid URLs found in CSV")

        # Extract features
        print("Extracting features from URLs...")
        X = np.array([self.feature_extractor.extract_features(url) for url in urls])

        # Scale features
        print("Scaling features...")
        X_scaled = self.scaler.fit_transform(X)

        # Optional: Split data for validation (80/20)
        if len(urls) > 100:
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        else:
            X_train, y_train = X_scaled, y
            X_test, y_test = None, None
            print(f"Training on {len(X_train)} samples (no test split for small dataset)")

        # Train model
        print("Training model...")
        self.model.fit(X_train, y_train)

        # Evaluate if test set exists
        if X_test is not None:
            train_score = self.model.score(X_train, y_train)
            test_score = self.model.score(X_test, y_test)
            print(f"Training accuracy: {train_score:.4f}")
            print(f"Test accuracy: {test_score:.4f}")

        # Save model
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def _train_synthetic_model(self):
        """Train model on synthetic data based on rules (fallback)"""
        print("Training on synthetic data...")
        # Generate synthetic examples
        synthetic_urls = [
            # Malicious examples
            "http://malware.example.com/download.exe",
            "https://192.168.1.1/fake-login.php",
            "http://scam-site.tk/win-prize",
            "https://bit.ly/suspicious-link",
            "http://phishing-site.ga/login",
            "https://virus-download.com/crack.exe",
            "http://fraud-site.cf/steal-data",
            "https://hack-tool.xyz/exploit",

            # Benign examples
            "https://www.google.com",
            "https://github.com/user/repo",
            "https://stackoverflow.com/questions",
            "https://www.wikipedia.org",
            "https://www.microsoft.com",
            "https://www.apple.com",
            "https://www.amazon.com",
            "https://www.facebook.com",
        ]

        # Generate more synthetic examples
        for i in range(20):
            synthetic_urls.append(f"https://legitimate-site-{i}.com/page")
            synthetic_urls.append(f"http://malicious-{i}.tk/download")

        # Extract features
        X = np.array([self.feature_extractor.extract_features(url) for url in synthetic_urls])

        # Create labels (1 = malicious, 0 = benign)
        y = np.array([
            1 if any(keyword in url.lower() for keyword in
                    ['malware', 'scam', 'phishing', 'virus', 'fraud', 'hack',
                     '192.168', 'bit.ly', '.tk', '.ga', '.cf', '.xyz', 'crack', 'exploit'])
            else 0
            for url in synthetic_urls
        ])

        # Scale features
        X_scaled = self.scaler.fit_transform(X)

        # Train model
        self.model.fit(X_scaled, y)

        # Save model
        try:
            joblib.dump({
                'model': self.model,
                'scaler': self.scaler
            }, self.model_path)
            print(f"Model saved to {self.model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")

    def predict(self, url: str) -> Dict[str, any]:
        """
        Predict if URL is malicious

        Returns:
            {
                'is_malicious': bool,
                'probability': float,
                'confidence': str
            }
        """
        if not url:
            return {
                'is_malicious': False,
                'probability': 0.0,
                'confidence': 'low'
            }

        # Extract features
        features = np.array([self.feature_extractor.extract_features(url)])

        # Scale features
        features_scaled = self.scaler.transform(features)

        # Predict
        probability = self.model.predict_proba(features_scaled)[0][1]  # Probability of being malicious
        is_malicious = probability > 0.5

        # Determine confidence
        if probability > 0.8:
            confidence = 'high'
        elif probability > 0.6:
            confidence = 'medium'
        else:
            confidence = 'low'

        return {
            'is_malicious': bool(is_malicious),
            'probability': float(probability),
            'confidence': confidence
        }

    def check_url(self, url: str) -> bool:
        """Simple check - returns True if URL is malicious"""
        result = self.predict(url)
        return result['is_malicious']


# Global instance
url_detector = MaliciousURLDetector()

