from .tactile_pointnet_concat import (
    TactilePointNetPlusPlusConcat,
    create_tactile_pointnet_concat_model,
)


def create_model(config):
    """Create the release contact-field model from config."""
    model_config = config.get('model', {})
    model_type = model_config.get('type', 'tactile_pointnet_concat')

    if model_type != 'tactile_pointnet_concat':
        raise ValueError(
            f"Unknown model type: {model_type}. "
            "This release keeps only 'tactile_pointnet_concat'."
        )

    print("Creating model: tactile_pointnet_concat")
    return create_tactile_pointnet_concat_model(config)


def get_model_info(config):
    model_type = config.get('model', {}).get('type', 'tactile_pointnet_concat')
    return {
        'model_type': model_type,
        'class_name': 'TactilePointNetPlusPlusConcat',
        'uses_environment': True,
        'architecture': 'concat',
        'description': 'TactilePointNet++ tactile-as-pointcloud model with concatenated tactile/object/environment points',
        'config_valid': model_type == 'tactile_pointnet_concat',
    }


__all__ = [
    'TactilePointNetPlusPlusConcat',
    'create_model',
    'get_model_info',
    'create_tactile_pointnet_concat_model',
]
