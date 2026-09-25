from photodoctor.ai.resources import AIResourcePlan, SystemResources, choose_ai_resource_plan

G = 1024 ** 3


def res(ram_gb, avail_gb, cpus=16, gpu_gb=None):
    return SystemResources(
        logical_cpus=cpus,
        total_ram_bytes=int(ram_gb * G),
        available_ram_bytes=int(avail_gb * G),
        cpu_name="test cpu",
        gpu_name="test gpu" if gpu_gb else None,
        gpu_vram_bytes=None if gpu_gb is None else int(gpu_gb * G),
    )


def test_5700x_16gb_class_gets_standard_profile():
    plan = choose_ai_resource_plan(res(16, 13, 16))
    assert plan.key == "standard"
    assert plan.patch_batch_size >= 16
    assert plan.max_surface_candidates >= 24
    assert plan.reserve_for_os_gib >= 5


def test_32gb_uses_larger_batch_than_16gb():
    p16 = choose_ai_resource_plan(res(16, 13, 16))
    p32 = choose_ai_resource_plan(res(32, 27, 16))
    assert p32.key == "extended"
    assert p32.patch_batch_size > p16.patch_batch_size
    assert p32.max_surface_candidates > p16.max_surface_candidates


def test_64gb_reserves_windows_memory_but_exposes_large_budget():
    plan = choose_ai_resource_plan(res(64, 58, 16))
    assert plan.key == "maximum"
    assert plan.reserve_for_os_gib >= 8
    assert plan.working_memory_budget_gib > 20
    assert plan.patch_batch_size >= 64


def test_low_current_free_memory_can_downgrade_even_large_machine():
    plan = choose_ai_resource_plan(res(64, 7, 16))
    assert plan.key == "conservative"
    assert plan.working_memory_budget_gib < 5


def test_gpu_is_not_required_for_standard_cpu_profile():
    plan = choose_ai_resource_plan(res(16, 13, 16, gpu_gb=None))
    assert plan.key == "standard"


def test_8_thread_cpu_without_smt_still_gets_standard_profile():
    plan = choose_ai_resource_plan(res(16, 13, 8))
    assert plan.key == "standard"
    assert plan.patch_batch_size >= 16
