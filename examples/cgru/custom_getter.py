# From [github]/tensorflow/python/kernel_tests/variable_scope_test.py
  def testGetterThatCreatesTwoVariablesAndSumsThem(self):

    def custom_getter(getter, name, *args, **kwargs):
      g_0 = getter("%s/0" % name, *args, **kwargs)
      g_1 = getter("%s/1" % name, *args, **kwargs)
      with tf.name_scope("custom_getter"):
        return g_0 + g_1  # or g_0 * const / ||g_0|| or anything you want

    with variable_scope.variable_scope("scope", custom_getter=custom_getter):
      v = variable_scope.get_variable("v", [1, 2, 3])
      # Or a full model if you wish. OO layers are ok.

    self.assertEqual([1, 2, 3], v.get_shape())
    true_vars = variables_lib.trainable_variables()
    self.assertEqual(2, len(true_vars))
    self.assertEqual("scope/v/0:0", true_vars[0].name)
    self.assertEqual("scope/v/1:0", true_vars[1].name)
    self.assertEqual("custom_getter/add:0", v.name)
    with self.test_session() as sess:
      variables_lib.global_variables_initializer().run()
      np_vars, np_v = sess.run([true_vars, v])
      self.assertAllClose(np_v, sum(np_vars))
