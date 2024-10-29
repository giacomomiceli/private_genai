# Sandbox script for querying the Weaviate instance directly using the REST API, without python wrapper.
# The REST API endpoints are documented here:
# https://weaviate.io/developers/weaviate/api/rest#description/introduction

$weaviateUrl = "http://localhost:8080/"

$endpoint = "$weaviateUrl/v1/objects/83ad54f0-813a-4587-b1d2-2927278a9e41/?include=vector"

$response = Invoke-RestMethod -Uri $endpoint -Method Get

Write-Host $response