import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, recall_score, precision_score
from sklearn.utils.class_weight import compute_class_weight

from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# ---------------------------
# Page config and theme
# ---------------------------
st.set_page_config(page_title="AI Classification", layout="wide")
st.markdown("""
<style>
.reportview-container .main .block-container{max-width: 1100px;}
h1, h2, h3 { font-weight: 600; }
.stMetric { background: #f7f9fc; padding: 12px; border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ---------------------------
# File upload functionality
# ---------------------------
st.title("AI Classification with Custom CSV Upload")

# Allow user to upload a CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)
    
    # Allow user to select the target column dynamically
    target_col = st.selectbox("Select the target column", options=df.columns)
    
    # Display a preview of the dataset
    st.write("Data uploaded successfully!")
    st.write("Data preview:", df.head())

    # Get all columns except the target column
    all_columns = [c for c in df.columns if c != target_col]

    # ---------------------------
    # Display basic information about the dataset
    # ---------------------------
    top_a, top_b, top_c = st.columns([2,2,1])
    with top_a:
        st.write("Original data dimensions:", df.shape)
    with top_b:
        st.write("Target distribution:", df[target_col].value_counts().sort_index())
    with top_c:
        st.write("Number of columns:", len(all_columns))

    # ---------------------------
    # Sidebar settings
    # ---------------------------
    st.sidebar.header("Training settings")
    cv_mode = st.sidebar.radio("Verification mode", ["Single cut (Train/Test)", "K-Fold Cross-validation"], index=0)
    random_state = st.sidebar.number_input("RANDOM_STATE", min_value=0, max_value=99999, value=22, step=1)
    test_size = st.sidebar.slider("test set ratio (Single cut)", 0.1, 0.5, 0.3, 0.05)
    n_splits = st.sidebar.slider("K-Fold Cross-validation", 3, 10, 5, 1)
    shuffle_data = st.sidebar.checkbox("Shuffle data", value=True)

    st.sidebar.subheader("Model Integration")
    use_voting = st.sidebar.checkbox("Voting Integration (XGB + RF + LR + SVC)", value=True)
    use_stacking = st.sidebar.checkbox("Stacked Integration (XGB + RF + LR -> SVC)", value=False)

    st.sidebar.subheader("XGBoost Parameters")
    learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, 0.01)
    max_depth = st.sidebar.slider("max_depth", 3, 12, 6, 1)
    n_estimators = st.sidebar.slider("n_estimators", 100, 1200, 500, 50)
    subsample = st.sidebar.slider("subsample", 0.5, 1.0, 0.8, 0.05)
    colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 0.8, 0.05)
    reg_lambda = st.sidebar.slider("reg_lambda (L2)", 0.0, 5.0, 1.0, 0.1)

    st.sidebar.subheader("Feature selection")
    selected_features = st.sidebar.multiselect("Features used", options=all_columns, default=all_columns[:5])

    st.sidebar.subheader("Handle imbalanced data")
    use_smote = st.sidebar.checkbox("Enable SMOTE (synthetic minority class samples)", value=True)

    run_btn = st.sidebar.button("Start training")

    # ---------------------------
    # Helper functions
    # ---------------------------

    def apply_smote(X_train, y_train, random_state):
        class_counts = np.bincount(y_train)
        min_class_size = min(class_counts)
        n_neighbors = min(6, min_class_size - 1) if min_class_size > 1 else 1
        
        smote = SMOTE(random_state=random_state, k_neighbors=n_neighbors)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        return X_train_resampled, y_train_resampled

    def make_xgb(num_classes:int):
        return xgb.XGBClassifier(
            objective="multi:softprob",   
            num_class=num_classes,
            random_state=random_state,
            eval_metric="mlogloss",
            learning_rate=learning_rate,
            max_depth=max_depth,
            n_estimators=n_estimators,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            use_label_encoder=False
        )

    def plot_confusion(cm, title="Confusion Matrix"):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        st.pyplot(fig)

    # ---------------------------
    # Training
    # ---------------------------
    if run_btn:
        if len(selected_features) == 0:
            st.warning("Please select at least one feature from the sidebar.")
            st.stop()

        X = df[selected_features].copy()
        y = df[target_col].copy()

        # Label encode
        label_encoder = LabelEncoder()
        y_enc = label_encoder.fit_transform(y)
        classes = np.unique(y_enc)
        num_classes = len(classes)

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Apply SMOTE to balance the classes if enabled
        if use_smote:
            X_scaled, y_enc = apply_smote(X_scaled, y_enc, random_state)

        # Class weights
        class_weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_enc)
        class_weights_dict = dict(enumerate(class_weights))

        # Base models
        rf = RandomForestClassifier(
            n_estimators=300, max_depth=None, random_state=random_state,
            class_weight="balanced_subsample", n_jobs=-1
        )
        lr = LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=random_state, multi_class="auto"
        )
        svc = SVC(
            probability=True, class_weight="balanced", random_state=random_state
        )

        voters = [
            ("xgb", make_xgb(num_classes)),
            ("rf", rf),
            ("lr", lr),
            ("svc", svc),
        ]
        if use_voting and use_stacking:
            st.info("Voting and Stacking cannot be enabled simultaneously; Stacking is prioritized.")
            use_voting = False

        if use_stacking:
            ensemble = StackingClassifier(
                estimators=voters[:-1],  # XGB, RF, LR
                final_estimator=svc,
                passthrough=True,
                n_jobs=None
            )
            model_name = "Stacking Ensemble (XGB + RF + LR -> SVC)"
        elif use_voting:
            ensemble = VotingClassifier(estimators=voters, voting="soft", n_jobs=None)
            model_name = "Voting Ensemble (XGB + RF + LR + SVC)"
        else:
            ensemble = make_xgb(num_classes)
            model_name = "XGBoost Single Model"

        st.subheader(f"Model: {model_name}")
        progress = st.progress(0)

        if cv_mode == "Single cut (Train/Test)":
            # Single split
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y_enc, test_size=test_size, shuffle=shuffle_data, random_state=random_state
            )

            # Apply SMOTE to the training set
            X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state)

            sample_weights = np.array([class_weights_dict[label] for label in y_train_resampled]) \
                if isinstance(ensemble, xgb.XGBClassifier) else None

            if isinstance(ensemble, xgb.XGBClassifier):
                ensemble.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights, eval_set=[(X_test, y_test)], verbose=False)
            else:
                ensemble.fit(X_train_resampled, y_train_resampled)

            y_pred = ensemble.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            f1m = f1_score(y_test, y_pred, average="macro")
            recall = recall_score(y_test, y_pred, average="macro")  # Average recall
            precision = precision_score(y_test, y_pred, average="macro")  # Average precision
            cm = confusion_matrix(y_test, y_pred)

            # Display results
            st.subheader("📊 Overall result")
            col_top1, col_top2 = st.columns(2)
            col_top1.metric("ACC", f"{acc:.4f}")
            col_top2.metric("F1-macro", f"{f1m:.4f}")
            st.metric("Average Recall", f"{recall:.4f}")
            st.metric("Average Precision", f"{precision:.4f}")

            progress.progress(100)
            st.write("Classification Report")
            st.text(classification_report(y_test, y_pred))

            # Display confusion matrix as a dataframe
            cm_df = pd.DataFrame(cm, index=label_encoder.classes_, columns=label_encoder.classes_)
            st.subheader("Confusion Matrix")
            st.dataframe(cm_df)

            plot_confusion(cm, title="Confusion Matrix")

        else:
            # K-Fold CV
            kf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle_data, random_state=random_state)
            acc_scores, f1_scores, recall_scores, precision_scores = [], [], [], []
            fold_idx = 1

            for i, (train_idx, test_idx) in enumerate(kf.split(X_scaled, y_enc)):
                X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
                y_train, y_test = y_enc[train_idx], y_enc[test_idx]

                # Apply SMOTE to the training set for each fold
                X_train_resampled, y_train_resampled = apply_smote(X_train, y_train, random_state)

                sample_weights = np.array([class_weights_dict[label] for label in y_train_resampled]) \
                    if isinstance(ensemble, xgb.XGBClassifier) else None

                if isinstance(ensemble, xgb.XGBClassifier):
                    ensemble.fit(X_train_resampled, y_train_resampled, sample_weight=sample_weights, eval_set=[(X_test, y_test)], verbose=False)
                else:
                    ensemble.fit(X_train_resampled, y_train_resampled)

                y_pred = ensemble.predict(X_test)
                acc = accuracy_score(y_test, y_pred)
                f1m = f1_score(y_test, y_pred, average="macro")
                recall = recall_score(y_test, y_pred, average="macro")  # Average recall
                precision = precision_score(y_test, y_pred, average="macro")  # Average precision
                acc_scores.append(acc)
                f1_scores.append(f1m)
                recall_scores.append(recall)
                precision_scores.append(precision)

                st.write(f"Fold {fold_idx} | ACC: {acc:.4f} | F1-macro: {f1m:.4f}")
                st.write(f"Fold {fold_idx} | Average Recall: {recall:.4f} | Average Precision: {precision:.4f}")

                cm = confusion_matrix(y_test, y_pred)
                st.write(f"Classification Report for Fold {fold_idx}")
                st.text(classification_report(y_test, y_pred))

                plot_confusion(cm, title=f"Fold {fold_idx} Confusion Matrix")

                fold_idx += 1
                progress.progress(int(100 * (i + 1) / n_splits))

            # Overall result
            st.subheader("📊 Overall result")
            col_r1, col_r2 = st.columns(2)
            col_r1.metric("Average ACC", f"{np.mean(acc_scores):.4f}")
            col_r2.metric("Average F1-macro", f"{np.mean(f1_scores):.4f}")
            st.metric("Average Recall", f"{np.mean(recall_scores):.4f}")
            st.metric("Average Precision", f"{np.mean(precision_scores):.4f}")

            st.write("Each fold ACC:", [round(a, 4) for a in acc_scores])
            st.write("Each fold F1-macro:", [round(f, 4) for f in f1_scores])
            st.write("Each fold Average Recall:", [round(r, 4) for r in recall_scores])
            st.write("Each fold Average Precision:", [round(p, 4) for p in precision_scores])

        # Feature importance (only available for XGB or voting models with XGB)
        if hasattr(ensemble, "feature_importances_"):
            st.subheader("Feature Importance")
            importance = ensemble.feature_importances_
            order = np.argsort(importance)
            fig2, ax2 = plt.subplots(figsize=(6, max(3, len(selected_features) * 0.35)))
            ax2.barh(np.array(selected_features)[order], importance[order])
            ax2.set_xlabel("Importance")
            st.pyplot(fig2)
        elif use_voting or use_stacking:
            st.info("The overall feature importance of the ensemble model may not be available; individual models' importance can be viewed.")
