PREFIX ?= $(HOME)/.local
PRODUCTS := build/Build/Products/Release
XCODEBUILD := xcodebuild -scheme slurper -destination platform=macOS,arch=arm64,variant=macos -derivedDataPath build \
	-skipPackagePluginValidation -skipMacroValidation

.DEFAULT_GOAL := help
.PHONY: help build test install model benchmark clean

help: ## List targets
	@awk 'BEGIN { FS = ":.*## " } /^[a-z]+:.*## / { printf "  %-10s %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

build: ## Release build in build/Build/Products/Release
	$(XCODEBUILD) -configuration Release -quiet build

test: ## Unit tests with a coverage report
	rm -rf build/Tests.xcresult
	$(XCODEBUILD) -enableCodeCoverage YES -resultBundlePath build/Tests.xcresult -quiet build test
	xcrun xccov view --report --json build/Tests.xcresult | python3 scripts/coverage_report.py

install: build ## Build and install to ~/.local/bin (override with PREFIX=...)
	mkdir -p $(PREFIX)/bin
	cp $(PRODUCTS)/slurper $(PREFIX)/bin/slurper

model: ## Rebuild the Core ML models from PyTorch into ~/Library/Application Support/Slurper/Models (needs uv)
	uv run scripts/convert_melband_roformer.py
	uv run scripts/convert_bs_roformer.py
	uv run scripts/convert_htdemucs.py

benchmark: build ## Score slurper against PyTorch demucs on the MUSDB18 test previews (needs uv)
	uv run scripts/benchmark.py $(PRODUCTS)/slurper build/benchmark

clean: ## Delete build products
	rm -rf build
