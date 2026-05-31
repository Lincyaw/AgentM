package wire

import (
	"crypto/tls"
	"net/http"
)

// newTLSHTTPClient creates an *http.Client with custom TLS configuration.
func newTLSHTTPClient(tlsConfig *tls.Config) *http.Client {
	return &http.Client{
		Transport: &http.Transport{
			TLSClientConfig: tlsConfig,
		},
	}
}
