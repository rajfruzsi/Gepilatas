history = model.fit(X_train, Y_train,
          batch_size=64, epochs=10,
          verbose=2,
          validation_data=(X_test, Y_test))
