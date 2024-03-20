class Testing():
    def run_tests(self, preprocessor, create_model, num_epochs,
                  train, split_and_fold_tests=False, filter_noise_tests=False, 
                  structure_clip_and_weighted_loss_tests=False, different_weighted_loss_tests=False,
                  additive_weighted_loss_tests=False, additional_tags=[]):
        preprocessing_configs = self._generate_configurations(train, split_and_fold_tests, 
                                                              filter_noise_tests, structure_clip_and_weighted_loss_tests,
                                                              different_weighted_loss_tests, additive_weighted_loss_tests,
                                                              additional_tags)
        
        # TODO: Test different min weight settings, eg. 0.1, 0.5
        for pre_config, tagging in preprocessing_configs:
            train_sets, preprocessing_config = preprocessor.prepare_xy_split(*pre_config)

            for experiment_type, train_data_loader, validation_data_loader in train_sets:
                model = create_model(experiment_type, num_epochs)

                model.fit(
                    train_data_loader,
                    validation_data_loader,
                    experiment_type=experiment_type,
                    epochs=num_epochs,
                    verbose=True,
                    preprocessing_config=preprocessing_config,
                    tags=tagging)
                del model
            
            del train_sets
            del preprocessing_config

    def _generate_configurations(self, train, split_and_fold_tests=False, 
                                filter_noise_tests=False, structure_clip_and_weighted_loss_tests=False,
                                different_weighted_loss_tests=False, additive_weighted_loss_tests=False,
                                additional_tags=[]):
        # Preprocessing testing
        # Dual model testing manually after all other tested
        preprocessing_configs = []
        tags = ['preprocessing evaluation'] + additional_tags

        # Keep fixed for consistency
        batch_size = 128
        shuffle = True

        # Manually test aferwards
        filter_noise = True
        dual_model = False

        if split_and_fold_tests:
            for validation_split, k_fold in [(0.1, None), (0.2, None), (None, 10), (None, 5)]:
                categorical = True
                structure = False
                clip = False
                weighted_loss = None
                additive_weight = False

                preprocessing_configs.append([[
                    train, categorical, shuffle, validation_split, 
                    batch_size, filter_noise, dual_model, 
                    k_fold, structure, clip, weighted_loss, additive_weight], tags + ['split and fold']])

        if filter_noise_tests:
            for filter_noise in [True, False]:
                categorical = True
                validation_split = None
                k_fold = 5
                weighted_loss = None
                additive_weight = False
                structure = False
                clip = False

                preprocessing_configs.append([[
                    train, categorical, shuffle, validation_split, 
                    batch_size, filter_noise, dual_model, 
                    k_fold, structure, clip, weighted_loss, additive_weight], tags + ['filter noise']])
            filter_noise = True

        if structure_clip_and_weighted_loss_tests:         
            for structure in [True, False]:
                for clip in [True, False]:
                    categorical = True
                    validation_split = None
                    k_fold = 5
                    additive_weight = False
                    weighted_loss = None

                    preprocessing_configs.append([[
                        train, categorical, shuffle, validation_split, 
                        batch_size, filter_noise, dual_model, 
                        k_fold, structure, clip, weighted_loss, additive_weight], tags + ['structure clip']])
                    
        if different_weighted_loss_tests:
            for weighted_loss in [None, 0.1, 0.3, 0.5, 0.7, 0.9]:
                categorical = True
                validation_split = None
                k_fold = 5
                structure = True
                clip = True
                additive_weight = False

                preprocessing_configs.append([[
                    train, categorical, shuffle, validation_split, 
                    batch_size, filter_noise, dual_model, 
                    k_fold, structure, clip, weighted_loss, additive_weight], tags + ['different weighted loss']])
                
        if additive_weighted_loss_tests:
            for weighted_loss in [None, 0.1, 0.3, 0.5, 0.7, 0.9]:
                categorical = True
                validation_split = None
                k_fold = 5
                structure = True
                clip = True
                additive_weight = True

                preprocessing_configs.append([[
                    train, categorical, shuffle, validation_split, 
                    batch_size, filter_noise, dual_model, 
                    k_fold, structure, clip, weighted_loss, additive_weight], tags + ['different additive weighted loss']])

        return preprocessing_configs